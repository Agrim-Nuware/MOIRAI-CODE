import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

SIZE = "base"
PDT = 20  # prediction length: forecast next 20 trading days (~1 month)
CTX = 300  # context length: trailing window fed to the model
PSZ = 16  # patch size for moirai-moe
BSZ = 32
TEST = 60  # holdout length for rolling evaluation

df = pd.read_csv("dataset/sp500/sp500_close_3y.csv", index_col=0, parse_dates=True)
df = df.asfreq("B")  # business-day frequency, required by gluonts
df["SP500"] = df["SP500"].ffill()

ds = PandasDataset(dict(df))

train, test_template = split(ds, offset=-TEST)
test_data = test_template.generate_instances(
    prediction_length=PDT,
    windows=TEST // PDT,
    distance=PDT,
)

model = MoiraiMoEForecast(
    module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}"),
    prediction_length=PDT,
    context_length=CTX,
    patch_size=PSZ,
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)

predictor = model.create_predictor(batch_size=BSZ)
forecasts = list(predictor.predict(test_data.input))
labels = list(test_data.label)
inputs = list(test_data.input)

print(f"Number of rolling windows: {len(forecasts)}")

maes, mapes = [], []
for inp, label, forecast in zip(inputs, labels, forecasts):
    actual = label["target"]
    pred = forecast.mean
    mae = np.mean(np.abs(actual - pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    maes.append(mae)
    mapes.append(mape)
    print(f"window: MAE={mae:.2f}  MAPE={mape:.2f}%")

print(f"\nAverage MAE across windows: {np.mean(maes):.2f}")
print(f"Average MAPE across windows: {np.mean(mapes):.2f}%")

fig, axes = plt.subplots(len(forecasts), 1, figsize=(10, 4 * len(forecasts)))
if len(forecasts) == 1:
    axes = [axes]
for ax, inp, label, forecast in zip(axes, inputs, labels, forecasts):
    plt.sca(ax)
    plot_single(
        inp,
        label,
        forecast,
        context_length=100,
        name="SP500 zero-shot forecast (Moirai-MoE-base)",
        show_label=True,
    )
plt.tight_layout()
plt.savefig("dataset/sp500/forecast_result.png", dpi=150)
print("\nSaved plot to dataset/sp500/forecast_result.png")
