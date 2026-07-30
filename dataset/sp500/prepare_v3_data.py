"""
v3 data prep: same 20-year S&P 500 download, engineered features, and
2024-10-01 date-based split as prepare_oct2024_split_data.py, but this time
every engineered variable is actually fed to the model -- as a context-only
covariate (`past_feat_dynamic_real`), never as something the model has to
forecast.

Target variates (forecasted): Close, Return, LogReturn.
Covariates (context only, never forecasted): Open, High, Low, Volume,
Momentum, MovingAverage, and cyclical (sin/cos) encodings of Day-of-week and
Month-of-year -- a plain integer 0-4/1-12 encoding would falsely tell the
model that e.g. December (12) is far from January (1), so day/month are
encoded as points on a circle instead (this is only needed for the numeric
model-input CSVs; the human-readable "Monday"/"January" names are kept in the
full reference CSV for your own inspection).

Writes:
  - sp500_v3_full_reference.csv        : full 20y, every engineered column,
                                          human-readable Day/Month names, for
                                          your own inspection/plotting.
  - sp500_v3_model_input_trainval.csv  : fine-tune pool only (< CUTOFF_DATE),
                                          numeric target + covariate columns
                                          only -- fed to uni2ts's data builder.
  - sp500_v3_model_input_full.csv      : same numeric columns, full 20y --
                                          used by the evaluation script's
                                          backtest (needs full history leading
                                          into the held-out region).
  - split_info_v3.json                 : lengths/offsets for the Hydra CLI.
"""

import json

import numpy as np
import pandas as pd
import yfinance as yf

TICKER = "^GSPC"
YEARS = "20y"
CUTOFF_DATE = "2024-10-01"  # fine-tune on everything before this, test on everything after
LIGHTNING_VAL_DAYS = 252  # tail of the pre-cutoff pool reserved for monitoring/checkpointing

MOMENTUM_WINDOW = 10
MA_WINDOW = 20

TARGET_COLUMNS = ["Close", "Return", "LogReturn"]
COVARIATE_COLUMNS = [
    "Open", "High", "Low", "Volume", "Momentum", "MovingAverage",
    "DaySin", "DayCos", "MonthSin", "MonthCos",
]
FULL_VARIATES = [
    "Open", "High", "Low", "Close", "Volume",
    "Day", "Month", "Momentum", "MovingAverage", "Return", "LogReturn",
]

OUT_DIR = "dataset/sp500"

df = yf.download(TICKER, period=YEARS, interval="1d", auto_adjust=True)
df.columns = df.columns.get_level_values(0)  # drop the ticker level
df = df[["Open", "High", "Low", "Close", "Volume"]]
df = df.asfreq("B")
df = df.ffill()

dayofweek = df.index.dayofweek  # 0=Mon .. 4=Fri (business days only)
month = df.index.month  # 1-12

df["Day"] = df.index.day_name()
df["Month"] = df.index.month_name()
df["Momentum"] = df["Close"] - df["Close"].shift(MOMENTUM_WINDOW)
df["MovingAverage"] = df["Close"].rolling(MA_WINDOW).mean()
df["Return"] = df["Close"].pct_change()
df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))

# Cyclical (sin/cos) encodings for the model -- these are what actually get
# fed in as covariates, not the human-readable Day/Month name columns above.
df["DaySin"] = np.sin(2 * np.pi * dayofweek / 5)
df["DayCos"] = np.cos(2 * np.pi * dayofweek / 5)
df["MonthSin"] = np.sin(2 * np.pi * (month - 1) / 12)
df["MonthCos"] = np.cos(2 * np.pi * (month - 1) / 12)

# Drop the warm-up rows where the rolling/shift features are NaN (max lookback
# is MA_WINDOW=20 rows) -- immaterial for a 20-year series.
df = df.dropna()

cutoff = pd.Timestamp(CUTOFF_DATE)
finetune_pool = df[df.index < cutoff]
final_test = df[df.index >= cutoff]

pool_len = len(finetune_pool)
train_length = pool_len - LIGHTNING_VAL_DAYS
final_test_len = len(final_test)

full_reference_path = f"{OUT_DIR}/sp500_v3_full_reference.csv"
model_input_trainval_path = f"{OUT_DIR}/sp500_v3_model_input_trainval.csv"
model_input_full_path = f"{OUT_DIR}/sp500_v3_model_input_full.csv"

model_columns = TARGET_COLUMNS + COVARIATE_COLUMNS

df[FULL_VARIATES].to_csv(full_reference_path)
finetune_pool[model_columns].to_csv(model_input_trainval_path)
df[model_columns].to_csv(model_input_full_path)

print(f"Total rows (20y, business days): {len(df)}")
print(f"Date range: {df.index[0].date()} -> {df.index[-1].date()}")
print(f"Fine-tune pool (< {CUTOFF_DATE}): {pool_len} rows, "
      f"{finetune_pool.index[0].date()} -> {finetune_pool.index[-1].date()}")
print(f"  train_length             = {train_length}")
print(f"  lightning_val_offset     = {train_length}")
print(f"  lightning_val_length     = {LIGHTNING_VAL_DAYS}")
print(f"Final test (>= {CUTOFF_DATE}): {final_test_len} rows, "
      f"{final_test.index[0].date()} -> {final_test.index[-1].date()}")
print(f"Target columns (forecasted): {TARGET_COLUMNS}")
print(f"Covariate columns (context only, never forecasted): {COVARIATE_COLUMNS}")

split_info = {
    "total_len": len(df),
    "cutoff_date": CUTOFF_DATE,
    "pool_len": pool_len,
    "train_length": train_length,
    "lightning_val_offset": train_length,
    "lightning_val_length": LIGHTNING_VAL_DAYS,
    "final_test_len": final_test_len,
    "target_columns": TARGET_COLUMNS,
    "covariate_columns": COVARIATE_COLUMNS,
    "full_variates": FULL_VARIATES,
    "momentum_window": MOMENTUM_WINDOW,
    "ma_window": MA_WINDOW,
}
with open(f"{OUT_DIR}/split_info_v3.json", "w") as f:
    json.dump(split_info, f, indent=2, default=str)

print(f"\nSplit info saved to {OUT_DIR}/split_info_v3.json")
