"""
Compare zero-shot vs fine-tuned Moirai-MoE-base on the held-out region on/after
CUTOFF_DATE (2024-10-01 by default -- see prepare_v3_data.py), same as v2, but
this time the model is also given every engineered variable as a context-only
covariate (past_feat_dynamic_real): Open, High, Low, Volume, Momentum,
MovingAverage, and cyclical Day/Month encodings. Only Close/Return/LogReturn
are ever forecasted -- the covariates inform the prediction without being
predicted themselves.

Produces:
  - dataset/sp500/results_v3_metrics.json
  - dataset/sp500/results_v3_forecast_plot.png    actual vs zero-shot vs fine-tuned vs reconstructed (Close)
  - dataset/sp500/results_v3_metrics_bar.png       Close MAPE + Return/LogReturn MAE, zero-shot vs fine-tuned
  - dataset/sp500/results_v3_loss_curve.png        train/val loss curve (if a TensorBoard log is found)

Usage:
  python dataset/sp500/evaluate_finetuned_v3.py --checkpoint <path/to/*.ckpt>
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

OUT_DIR = "dataset/sp500"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    default=None,
    help="Path to the fine-tuned .ckpt file. If omitted, globs for the most "
    "recent checkpoint under outputs/finetune/.",
)
parser.add_argument("--full_csv", default=f"{OUT_DIR}/sp500_v3_model_input_full.csv")
parser.add_argument("--split_info", default=f"{OUT_DIR}/split_info_v3.json")
parser.add_argument("--context_length", type=int, default=512)
parser.add_argument("--prediction_length", type=int, default=32)
parser.add_argument("--patch_size", type=int, default=16)
parser.add_argument("--num_samples", type=int, default=100)
parser.add_argument(
    "--n_windows",
    type=int,
    default=None,
    help="Override number of held-out test windows (defaults to final_test_len // prediction_length).",
)
parser.add_argument(
    "--distance",
    type=int,
    default=None,
    help="Stride between windows (defaults to prediction_length, i.e. non-overlapping).",
)
parser.add_argument(
    "--eval_batch_size",
    type=int,
    default=2,
    help="Inference batch size. With 10 covariates added, each packed sequence "
    "is ~4x longer than v2's target-only sequences, and attention memory "
    "scales roughly with sequence length squared -- kept low by default to "
    "avoid CUDA OOM on a free-tier GPU. Raise if you have headroom to spare.",
)
args = parser.parse_args()

CONTEXT_LENGTH = args.context_length
PREDICTION_LENGTH = args.prediction_length
PATCH_SIZE = args.patch_size
NUM_SAMPLES = args.num_samples

with open(args.split_info) as f:
    split_info = json.load(f)
final_test_len = split_info["final_test_len"]
TARGET_COLUMNS = split_info["target_columns"]
COVARIATE_COLUMNS = split_info["covariate_columns"]

ckpt_path = args.checkpoint
if ckpt_path is None:
    candidates = sorted(
        glob.glob("outputs/finetune/**/*.ckpt", recursive=True),
        key=os.path.getmtime,
    )
    assert candidates, "No checkpoint found under outputs/finetune/ -- pass --checkpoint explicitly."
    ckpt_path = candidates[-1]
print(f"Using checkpoint: {ckpt_path}")

df = pd.read_csv(args.full_csv, index_col=0, parse_dates=True)[TARGET_COLUMNS + COVARIATE_COLUMNS]
df = df.asfreq("B").ffill()
print(f"Full series: {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")
print(f"Target columns: {TARGET_COLUMNS}")
print(f"Covariate columns: {COVARIATE_COLUMNS}")

ds = PandasDataset(
    dataframes=df,
    target=TARGET_COLUMNS,
    past_feat_dynamic_real=COVARIATE_COLUMNS,
    freq="B",
)

distance = args.distance if args.distance is not None else PREDICTION_LENGTH
n_windows = args.n_windows if args.n_windows is not None else final_test_len // PREDICTION_LENGTH
train, test_template = split(ds, offset=-final_test_len)
test_data = test_template.generate_instances(
    prediction_length=PREDICTION_LENGTH,
    windows=n_windows,
    distance=distance,
)
print(f"Held-out test (>= {split_info['cutoff_date']}): {n_windows} windows of {PREDICTION_LENGTH} days, stride {distance}")


def build_forecast_model(module: MoiraiMoEModule) -> MoiraiMoEForecast:
    return MoiraiMoEForecast(
        module=module,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        patch_size=PATCH_SIZE,
        num_samples=NUM_SAMPLES,
        target_dim=len(TARGET_COLUMNS),
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=len(COVARIATE_COLUMNS),
    )


def run_backtest(forecast_model: MoiraiMoEForecast):
    predictor = forecast_model.create_predictor(batch_size=args.eval_batch_size)
    forecasts = list(predictor.predict(test_data.input))
    labels = list(test_data.label)
    return forecasts, labels


import gc

print("\n--- Zero-shot (pretrained) ---")
zs_module = MoiraiMoEModule.from_pretrained("Salesforce/moirai-moe-1.0-R-base")
zs_forecasts, labels = run_backtest(build_forecast_model(zs_module))

# Release the zero-shot model's GPU memory before loading the second
# ~935M-param model -- otherwise both sit in VRAM simultaneously and a
# free-tier GPU runs out of memory during the fine-tuned model's inference.
del zs_module
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

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


close_idx = TARGET_COLUMNS.index("Close")
logreturn_idx = TARGET_COLUMNS.index("LogReturn")


def reconstruct_close_from_logreturn(logreturn_path, last_close):
    """Compound a forecasted log-return path onto the last known Close price."""
    return last_close * np.exp(np.cumsum(logreturn_path))


def compute_metrics(forecasts, labels, inputs):
    per_variate = {v: {"mae": []} for v in TARGET_COLUMNS}
    per_variate["Close"]["mape"] = []
    recon = {"mae": [], "mape": []}

    for forecast, label, inp in zip(forecasts, labels, inputs):
        actual_2d = np.asarray(label["target"])  # (target_dim, prediction_length)
        for i, v in enumerate(TARGET_COLUMNS):
            actual = actual_2d[i]
            pred = extract_multivariate(forecast, i)
            mae = float(np.mean(np.abs(actual - pred)))
            per_variate[v]["mae"].append(mae)
            if v == "Close":
                mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
                per_variate[v]["mape"].append(mape)

        last_close = np.asarray(inp["target"])[close_idx][-1]
        logreturn_pred = extract_multivariate(forecast, logreturn_idx)
        actual_close = actual_2d[close_idx]
        recon_close = reconstruct_close_from_logreturn(logreturn_pred, last_close)
        recon["mae"].append(float(np.mean(np.abs(actual_close - recon_close))))
        recon["mape"].append(float(np.mean(np.abs((actual_close - recon_close) / actual_close)) * 100))

    summary = {
        v: {k: float(np.mean(vals)) for k, vals in m.items()}
        for v, m in per_variate.items()
    }
    summary["Close_reconstructed"] = {k: float(np.mean(vals)) for k, vals in recon.items()}
    return summary


zs_metrics = compute_metrics(zs_forecasts, labels, test_data.input)
ft_metrics = compute_metrics(ft_forecasts, labels, test_data.input)

print("\n=== Results (avg over test windows) ===")
print(f"{'variate':<20} {'zero-shot':>18} {'fine-tuned':>18}")
print(f"{'Close MAPE':<20} {zs_metrics['Close']['mape']:>17.2f}% {ft_metrics['Close']['mape']:>17.2f}%")
print(f"{'Close(recon) MAPE':<20} {zs_metrics['Close_reconstructed']['mape']:>17.2f}% {ft_metrics['Close_reconstructed']['mape']:>17.2f}%")
print(f"{'Return MAE':<20} {zs_metrics['Return']['mae']:>18.5f} {ft_metrics['Return']['mae']:>18.5f}")
print(f"{'LogReturn MAE':<20} {zs_metrics['LogReturn']['mae']:>18.5f} {ft_metrics['LogReturn']['mae']:>18.5f}")

results = {"zero_shot": zs_metrics, "fine_tuned": ft_metrics, "checkpoint": ckpt_path}
with open(f"{OUT_DIR}/results_v3_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

# --- Plot 1: forecast comparison on Close price (direct + reconstructed) ---
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
    last_close = np.asarray(inp["target"])[close_idx][-1]
    pred_ft_recon = reconstruct_close_from_logreturn(
        extract_multivariate(forecast_ft, logreturn_idx), last_close
    )
    start = forecast_zs.start_date.to_timestamp()
    fcst_index = pd.date_range(start, periods=PREDICTION_LENGTH, freq="B")

    context_tail = 60
    hist_target = np.asarray(inp["target"])[close_idx][-context_tail:]
    hist_index = pd.date_range(end=start - pd.tseries.frequencies.to_offset("B"), periods=context_tail, freq="B")

    ax.plot(hist_index, hist_target, color="black", label="history (Close)")
    ax.plot(fcst_index, actual_future, color="black", linestyle="--", label="actual")
    ax.plot(fcst_index, pred_zs, color="tab:orange", label="zero-shot (direct)")
    ax.plot(fcst_index, pred_ft, color="tab:blue", label="fine-tuned (direct)")
    ax.plot(fcst_index, pred_ft_recon, color="tab:green", linestyle=":", label="fine-tuned (from LogReturn)")
    ax.set_title(f"Window starting {fcst_index[0].date()}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/results_v3_forecast_plot.png", dpi=150)
print(f"\nSaved {OUT_DIR}/results_v3_forecast_plot.png")

# --- Plot 2: metrics bar charts ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

close_labels = ["Zero-shot\n(direct)", "Fine-tuned\n(direct)", "Fine-tuned\n(from LogReturn)"]
close_values = [
    zs_metrics["Close"]["mape"],
    ft_metrics["Close"]["mape"],
    ft_metrics["Close_reconstructed"]["mape"],
]
ax1.bar(close_labels, close_values, color=["tab:orange", "tab:blue", "tab:green"])
ax1.set_ylabel("MAPE (%)")
ax1.set_title("Close price forecast error")

x = np.arange(2)
width = 0.35
ax2.bar(x - width / 2, [zs_metrics["Return"]["mae"], zs_metrics["LogReturn"]["mae"]], width, label="Zero-shot", color="tab:orange")
ax2.bar(x + width / 2, [ft_metrics["Return"]["mae"], ft_metrics["LogReturn"]["mae"]], width, label="Fine-tuned", color="tab:blue")
ax2.set_xticks(x)
ax2.set_xticklabels(["Return", "LogReturn"])
ax2.set_ylabel("MAE")
ax2.set_title("Return / LogReturn forecast error")
ax2.legend()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/results_v3_metrics_bar.png", dpi=150)
print(f"Saved {OUT_DIR}/results_v3_metrics_bar.png")

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
        plt.savefig(f"{OUT_DIR}/results_v3_loss_curve.png", dpi=150)
        print(f"Saved {OUT_DIR}/results_v3_loss_curve.png")
    else:
        print("No TensorBoard log found, skipping loss curve plot.")
except Exception as e:
    print(f"Could not plot loss curve: {e}")

print("\nDone.")
