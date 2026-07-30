"""
Compare zero-shot vs fine-tuned Moirai-MoE-base on the held-out final test
region (~1 trading year, never seen during data building, training, or
Lightning validation).

Produces:
  - dataset/sp500/results_metrics.json          MAE/MAPE per variate, both models
  - dataset/sp500/results_forecast_plot.png     actual vs zero-shot vs fine-tuned (Close)
  - dataset/sp500/results_metrics_bar.png        MAPE bar chart, zero-shot vs fine-tuned
  - dataset/sp500/results_loss_curve.png         train/val loss curve (if a
                                                   TensorBoard log is found)

Usage:
  python dataset/sp500/evaluate_finetuned.py --checkpoint <path/to/*.ckpt>
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

VARIATES = ["Open", "High", "Low", "Close", "Volume"]
OUT_DIR = "dataset/sp500"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    default=None,
    help="Path to the fine-tuned .ckpt file. If omitted, globs for the most "
    "recent checkpoint under outputs/finetune/.",
)
parser.add_argument("--full_csv", default=f"{OUT_DIR}/sp500_20y_full.csv")
parser.add_argument("--split_info", default=f"{OUT_DIR}/split_info.json")
parser.add_argument("--context_length", type=int, default=512)
parser.add_argument("--prediction_length", type=int, default=32)
parser.add_argument("--patch_size", type=int, default=16)
parser.add_argument("--num_samples", type=int, default=100)
parser.add_argument(
    "--n_windows",
    type=int,
    default=None,
    help="Override number of held-out test windows (defaults to final_test_days // prediction_length).",
)
args = parser.parse_args()

CONTEXT_LENGTH = args.context_length
PREDICTION_LENGTH = args.prediction_length
PATCH_SIZE = args.patch_size
NUM_SAMPLES = args.num_samples

with open(args.split_info) as f:
    split_info = json.load(f)
final_test_days = split_info["final_test_days"]

ckpt_path = args.checkpoint
if ckpt_path is None:
    candidates = sorted(
        glob.glob("outputs/finetune/**/*.ckpt", recursive=True),
        key=os.path.getmtime,
    )
    assert candidates, "No checkpoint found under outputs/finetune/ -- pass --checkpoint explicitly."
    ckpt_path = candidates[-1]
print(f"Using checkpoint: {ckpt_path}")

df = pd.read_csv(args.full_csv, index_col=0, parse_dates=True)
df = df.asfreq("B").ffill()
print(f"Full series: {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")

ds = PandasDataset(dataframes=df, target=VARIATES, freq="B")

n_windows = args.n_windows if args.n_windows is not None else final_test_days // PREDICTION_LENGTH
train, test_template = split(ds, offset=-final_test_days)
test_data = test_template.generate_instances(
    prediction_length=PREDICTION_LENGTH,
    windows=n_windows,
    distance=PREDICTION_LENGTH,
)
print(f"Held-out final test: {n_windows} non-overlapping windows of {PREDICTION_LENGTH} days")


def build_forecast_model(module: MoiraiMoEModule) -> MoiraiMoEForecast:
    return MoiraiMoEForecast(
        module=module,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        patch_size=PATCH_SIZE,
        num_samples=NUM_SAMPLES,
        target_dim=len(VARIATES),
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )


def run_backtest(forecast_model: MoiraiMoEForecast):
    predictor = forecast_model.create_predictor(batch_size=8)
    forecasts = list(predictor.predict(test_data.input))
    labels = list(test_data.label)
    return forecasts, labels


print("\n--- Zero-shot (pretrained) ---")
zs_module = MoiraiMoEModule.from_pretrained("Salesforce/moirai-moe-1.0-R-base")
zs_forecasts, labels = run_backtest(build_forecast_model(zs_module))

print("\n--- Fine-tuned ---")
ft_module = MoiraiMoEModule.from_pretrained("Salesforce/moirai-moe-1.0-R-base")
state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
module_state = {
    k[len("module.") :]: v for k, v in state_dict.items() if k.startswith("module.")
}
missing, unexpected = ft_module.load_state_dict(module_state, strict=True)
ft_forecasts, _ = run_backtest(build_forecast_model(ft_module))


def extract_multivariate(forecast, variate_idx):
    """GluonTS SampleForecast.mean for a multivariate (target_dim>1) forecast
    has shape (prediction_length, target_dim); slice out one variate's series."""
    mean = np.asarray(forecast.mean)
    if mean.shape[0] == PREDICTION_LENGTH:
        return mean[:, variate_idx]
    return mean[variate_idx]  # already (target_dim, prediction_length)


def compute_metrics(forecasts, labels):
    per_variate = {v: {"mae": [], "mape": []} for v in VARIATES}
    for forecast, label in zip(forecasts, labels):
        actual_2d = np.asarray(label["target"])  # (target_dim, prediction_length)
        for i, v in enumerate(VARIATES):
            actual = actual_2d[i]
            pred = extract_multivariate(forecast, i)
            mae = float(np.mean(np.abs(actual - pred)))
            mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
            per_variate[v]["mae"].append(mae)
            per_variate[v]["mape"].append(mape)
    summary = {
        v: {
            "mae": float(np.mean(m["mae"])),
            "mape": float(np.mean(m["mape"])),
        }
        for v, m in per_variate.items()
    }
    return summary


zs_metrics = compute_metrics(zs_forecasts, labels)
ft_metrics = compute_metrics(ft_forecasts, labels)

print("\n=== Results (avg over test windows) ===")
print(f"{'variate':<8} {'zero-shot MAPE':>16} {'fine-tuned MAPE':>16}")
for v in VARIATES:
    print(f"{v:<8} {zs_metrics[v]['mape']:>15.2f}% {ft_metrics[v]['mape']:>15.2f}%")

results = {"zero_shot": zs_metrics, "fine_tuned": ft_metrics, "checkpoint": ckpt_path}
with open(f"{OUT_DIR}/results_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

# --- Plot 1: forecast comparison on Close price ---
close_idx = VARIATES.index("Close")
n_plot = min(n_windows, 4)
fig, axes = plt.subplots(n_plot, 1, figsize=(11, 4 * n_plot))
if n_plot == 1:
    axes = [axes]

for ax, forecast_zs, forecast_ft, label, inp in zip(
    axes, zs_forecasts[:n_plot], ft_forecasts[:n_plot], labels[:n_plot], test_data.input
):
    actual_future = np.asarray(label["target"])[close_idx]
    pred_zs = extract_multivariate(forecast_zs, close_idx)
    pred_ft = extract_multivariate(forecast_ft, close_idx)
    start = forecast_zs.start_date.to_timestamp()
    fcst_index = pd.date_range(start, periods=PREDICTION_LENGTH, freq="B")

    context_tail = 60
    hist_target = np.asarray(inp["target"])[close_idx][-context_tail:]
    hist_index = pd.date_range(end=start - pd.tseries.frequencies.to_offset("B"), periods=context_tail, freq="B")

    ax.plot(hist_index, hist_target, color="black", label="history (Close)")
    ax.plot(fcst_index, actual_future, color="black", linestyle="--", label="actual")
    ax.plot(fcst_index, pred_zs, color="tab:orange", label="zero-shot forecast")
    ax.plot(fcst_index, pred_ft, color="tab:blue", label="fine-tuned forecast")
    ax.set_title(f"Window starting {fcst_index[0].date()}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/results_forecast_plot.png", dpi=150)
print(f"\nSaved {OUT_DIR}/results_forecast_plot.png")

# --- Plot 2: MAPE bar chart, all variates ---
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(VARIATES))
width = 0.35
ax.bar(x - width / 2, [zs_metrics[v]["mape"] for v in VARIATES], width, label="Zero-shot", color="tab:orange")
ax.bar(x + width / 2, [ft_metrics[v]["mape"] for v in VARIATES], width, label="Fine-tuned", color="tab:blue")
ax.set_xticks(x)
ax.set_xticklabels(VARIATES)
ax.set_ylabel("MAPE (%)")
ax.set_title("Zero-shot vs Fine-tuned Moirai-MoE-base -- S&P 500 held-out test year")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/results_metrics_bar.png", dpi=150)
print(f"Saved {OUT_DIR}/results_metrics_bar.png")

# --- Plot 3: training/validation loss curve, if a TensorBoard log exists ---
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb_dirs = glob.glob("outputs/finetune/**/logs/**/events.out.tfevents.*", recursive=True)
    if tb_dirs:
        tb_dir = os.path.dirname(sorted(tb_dirs, key=os.path.getmtime)[-1])
        ea = EventAccumulator(tb_dir)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        fig, ax = plt.subplots(figsize=(9, 5))
        for tag in tags:
            if "PackedNLLLoss" in tag:
                events = ea.Scalars(tag)
                ax.plot([e.step for e in events], [e.value for e in events], label=tag)
        ax.set_xlabel("step")
        ax.set_ylabel("PackedNLLLoss")
        ax.set_title("Fine-tuning loss curve")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/results_loss_curve.png", dpi=150)
        print(f"Saved {OUT_DIR}/results_loss_curve.png")
    else:
        print("No TensorBoard log found, skipping loss curve plot.")
except Exception as e:
    print(f"Could not plot loss curve: {e}")

print("\nDone.")
