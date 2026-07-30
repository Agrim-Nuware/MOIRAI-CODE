"""
Download 20 years of S&P 500 (^GSPC) data, engineer a feature set, and split
it chronologically at a fixed calendar date (not a fixed row count):

  [ FINE-TUNE POOL: everything before CUTOFF_DATE ] [ FINAL TEST: everything on/after CUTOFF_DATE ]

The fine-tune pool is further split into the actual training region and a
Lightning-validation tail (used only for monitoring / checkpoint selection,
never for gradient updates, and never overlapping the final test region).

Engineered columns (on top of raw Open/High/Low/Close/Volume):
  - Day            : day of week, 0=Mon .. 4=Fri (weekday seasonality)
  - Month          : calendar month, 1-12 (month-of-year seasonality)
  - Momentum       : Close - Close.shift(MOMENTUM_WINDOW)   (10-day price momentum)
  - MovingAverage  : Close.rolling(MA_WINDOW).mean()         (20-day SMA)
  - Return         : Close.pct_change()                      (simple daily return)
  - LogReturn      : log(Close / Close.shift(1))              (log daily return)

Only CONTEXT_VARIATES (Close, Return, LogReturn) are ever fed to the model as
target variates -- Open/High/Low/Volume/Day/Month/Momentum/MovingAverage are
computed and saved for reference/plotting only. uni2ts's simple multivariate
data builder has no notion of a "covariate-only, not forecasted" channel, so
"out of context" here means excluded from the model entirely, not just from
the forecast target.

Writes:
  - sp500_full_features.csv     : full 20y, every engineered column, used by
                                   the evaluation script (backtest needs full
                                   history + the held-out region).
  - sp500_context_trainval.csv  : fine-tune pool only (before CUTOFF_DATE),
                                   CONTEXT_VARIATES columns only -- this is
                                   what gets fed into uni2ts.data.builder.simple.
  - split_info.json             : all the lengths/offsets needed for the
                                   Hydra CLI overrides.
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

CONTEXT_VARIATES = ["Close", "Return", "LogReturn"]
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

df["Day"] = df.index.dayofweek.astype("float32")
df["Month"] = df.index.month.astype("float32")
df["Momentum"] = df["Close"] - df["Close"].shift(MOMENTUM_WINDOW)
df["MovingAverage"] = df["Close"].rolling(MA_WINDOW).mean()
df["Return"] = df["Close"].pct_change()
df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))

# Drop the warm-up rows where the rolling/shift features are NaN (max lookback
# is MA_WINDOW=20 rows) -- immaterial for a 20-year series.
df = df.dropna()
df = df[FULL_VARIATES]

cutoff = pd.Timestamp(CUTOFF_DATE)
finetune_pool = df[df.index < cutoff]
final_test = df[df.index >= cutoff]

pool_len = len(finetune_pool)
train_length = pool_len - LIGHTNING_VAL_DAYS
final_test_len = len(final_test)

full_path = f"{OUT_DIR}/sp500_full_features.csv"
trainval_path = f"{OUT_DIR}/sp500_context_trainval.csv"

df.to_csv(full_path)
finetune_pool[CONTEXT_VARIATES].to_csv(trainval_path)

print(f"Total rows (20y, business days): {len(df)}")
print(f"Date range: {df.index[0].date()} -> {df.index[-1].date()}")
print(f"Fine-tune pool (< {CUTOFF_DATE}): {pool_len} rows, "
      f"{finetune_pool.index[0].date()} -> {finetune_pool.index[-1].date()}")
print(f"  train_length             = {train_length}")
print(f"  lightning_val_offset     = {train_length}")
print(f"  lightning_val_length     = {LIGHTNING_VAL_DAYS}")
print(f"Final test (>= {CUTOFF_DATE}): {final_test_len} rows, "
      f"{final_test.index[0].date()} -> {final_test.index[-1].date()}")
print(f"Context variates fed to the model: {CONTEXT_VARIATES}")
print(f"Full feature columns (reference only): {FULL_VARIATES}")

split_info = {
    "total_len": len(df),
    "cutoff_date": CUTOFF_DATE,
    "pool_len": pool_len,
    "train_length": train_length,
    "lightning_val_offset": train_length,
    "lightning_val_length": LIGHTNING_VAL_DAYS,
    "final_test_len": final_test_len,
    "context_variates": CONTEXT_VARIATES,
    "full_variates": FULL_VARIATES,
    "momentum_window": MOMENTUM_WINDOW,
    "ma_window": MA_WINDOW,
}
with open(f"{OUT_DIR}/split_info_v2.json", "w") as f:
    json.dump(split_info, f, indent=2, default=str)

print(f"\nSplit info saved to {OUT_DIR}/split_info_v2.json")
