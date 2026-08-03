"""
v4 data prep: fine-tune on a basket of 4 major US equity indices instead of
just the S&P 500, plus real exogenous macro covariates (VIX, 10-year Treasury
yield). Rationale (see the v3 postmortem): fine-tuning on one single narrow
series gives a 935M-parameter model no independent series to learn
generalizable structure from -- every training window is still "the same
series," just shifted in time. This is the most likely reason v2 (full
fine-tune) and v3 (freeze_ffn + covariates + early stopping) converged to
nearly identical, zero-shot-losing results despite very different recipes.

Tickers:
  GSPC = S&P 500      (^GSPC) -- the only one actually evaluated/backtested
  IXIC = Nasdaq Composite (^IXIC)  -- training-diversity only
  DJI  = Dow Jones Industrial (^DJI)  -- training-diversity only
  RUT  = Russell 2000 (^RUT)  -- training-diversity only
  VIX  = CBOE Volatility Index (^VIX) -- shared macro covariate, all items
  TNX  = 10-Year Treasury Note Yield (^TNX) -- shared macro covariate, all items

Per-ticker target columns (forecasted, only for GSPC at eval time):
  Close, Return, LogReturn

Per-ticker covariate columns (context only, never forecasted):
  Open, High, Low, Volume, Momentum, MovingAverage,
  DaySin, DayCos, MonthSin, MonthCos

Shared covariate columns (context only, same value across every item):
  VIX_Close, TNX_Close

Writes:
  - sp500_v4_full_reference.csv        : full 20y, every ticker's engineered
                                          columns (human-readable Day/Month
                                          names included) + macro series, for
                                          your own inspection/plotting.
  - sp500_v4_model_input_trainval.csv  : fine-tune pool only (< CUTOFF_DATE),
                                          numeric target + covariate columns
                                          for all 4 tickers + shared macro
                                          columns -- fed to uni2ts's data
                                          builder (--tickers GSPC,IXIC,DJI,RUT).
  - sp500_v4_model_input_full.csv      : same numeric columns, full 20y --
                                          used by the evaluation script's
                                          GSPC-only backtest (needs full
                                          history leading into the held-out
                                          region).
  - split_info_v4.json                 : lengths/offsets/column lists for the
                                          Hydra CLI and evaluation script.
"""

import json

import numpy as np
import pandas as pd
import yfinance as yf

EQUITY_TICKERS = {
    "GSPC": "^GSPC",
    "IXIC": "^IXIC",
    "DJI": "^DJI",
    "RUT": "^RUT",
}
MACRO_TICKERS = {
    "VIX": "^VIX",
    "TNX": "^TNX",
}
EVAL_TICKER = "GSPC"  # the only one actually forecasted/backtested

YEARS = "20y"
CUTOFF_DATE = "2024-10-01"  # fine-tune on everything before this, test on everything after
LIGHTNING_VAL_DAYS = 252  # tail of the pre-cutoff pool reserved for monitoring/checkpointing

MOMENTUM_WINDOW = 10
MA_WINDOW = 20

TARGET_SUFFIXES = ["Close", "Return", "LogReturn"]
COVARIATE_SUFFIXES = [
    "Open", "High", "Low", "Volume", "Momentum", "MovingAverage",
    "DaySin", "DayCos", "MonthSin", "MonthCos",
]
SHARED_COVARIATE_COLUMNS = ["VIX_Close", "TNX_Close"]
FULL_REFERENCE_SUFFIXES = [
    "Open", "High", "Low", "Close", "Volume",
    "Day", "Month", "Momentum", "MovingAverage", "Return", "LogReturn",
]

OUT_DIR = "dataset/sp500"


def engineer_equity_features(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period=YEARS, interval="1d", auto_adjust=True)
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
    df["DaySin"] = np.sin(2 * np.pi * dayofweek / 5)
    df["DayCos"] = np.cos(2 * np.pi * dayofweek / 5)
    df["MonthSin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["MonthCos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return df


print("Downloading and engineering features for 4 equity indices...")
equity_frames = {}
for name, symbol in EQUITY_TICKERS.items():
    print(f"  {name} ({symbol})...")
    equity_frames[name] = engineer_equity_features(symbol)

print("Downloading shared macro covariates...")
macro_frames = {}
for name, symbol in MACRO_TICKERS.items():
    print(f"  {name} ({symbol})...")
    raw = yf.download(symbol, period=YEARS, interval="1d", auto_adjust=True)
    raw.columns = raw.columns.get_level_values(0)
    macro_frames[name] = raw["Close"].asfreq("B").ffill()

# Align everything onto a single shared business-day index (the intersection
# of all tickers' available dates -- equity indices should already share the
# same US trading calendar, but macro series can start on slightly different
# dates).
all_engineered_columns = sorted(
    set(FULL_REFERENCE_SUFFIXES) | set(COVARIATE_SUFFIXES) | set(TARGET_SUFFIXES)
)
combined = pd.DataFrame(index=equity_frames[EVAL_TICKER].index)
for name, df in equity_frames.items():
    prefixed = df[all_engineered_columns].add_prefix(f"{name}_")
    combined = combined.join(prefixed, how="inner")
for name, series in macro_frames.items():
    combined[f"{name}_Close"] = series
combined = combined.asfreq("B").ffill().dropna()

cutoff = pd.Timestamp(CUTOFF_DATE)
finetune_pool = combined[combined.index < cutoff]
final_test = combined[combined.index >= cutoff]

pool_len = len(finetune_pool)
train_length = pool_len - LIGHTNING_VAL_DAYS
final_test_len = len(final_test)

model_columns = (
    [f"{t}_{s}" for t in EQUITY_TICKERS for s in TARGET_SUFFIXES]
    + [f"{t}_{s}" for t in EQUITY_TICKERS for s in COVARIATE_SUFFIXES]
    + SHARED_COVARIATE_COLUMNS
)

full_reference_path = f"{OUT_DIR}/sp500_v4_full_reference.csv"
model_input_trainval_path = f"{OUT_DIR}/sp500_v4_model_input_trainval.csv"
model_input_full_path = f"{OUT_DIR}/sp500_v4_model_input_full.csv"

combined.to_csv(full_reference_path)
finetune_pool[model_columns].to_csv(model_input_trainval_path)
combined[model_columns].to_csv(model_input_full_path)

eval_target_columns = [f"{EVAL_TICKER}_{s}" for s in TARGET_SUFFIXES]
eval_covariate_columns = [
    f"{EVAL_TICKER}_{s}" for s in COVARIATE_SUFFIXES
] + SHARED_COVARIATE_COLUMNS

print(f"\nTotal rows (20y, business days): {len(combined)}")
print(f"Date range: {combined.index[0].date()} -> {combined.index[-1].date()}")
print(f"Fine-tune pool (< {CUTOFF_DATE}): {pool_len} rows, "
      f"{finetune_pool.index[0].date()} -> {finetune_pool.index[-1].date()}")
print(f"  train_length             = {train_length}")
print(f"  lightning_val_offset     = {train_length}")
print(f"  lightning_val_length     = {LIGHTNING_VAL_DAYS}")
print(f"Final test (>= {CUTOFF_DATE}): {final_test_len} rows, "
      f"{final_test.index[0].date()} -> {final_test.index[-1].date()}")
print(f"Tickers (training diversity): {list(EQUITY_TICKERS)}")
print(f"Eval ticker (only one backtested): {EVAL_TICKER}")
print(f"Eval target columns: {eval_target_columns}")
print(f"Eval covariate columns: {eval_covariate_columns}")

split_info = {
    "total_len": len(combined),
    "cutoff_date": CUTOFF_DATE,
    "pool_len": pool_len,
    "train_length": train_length,
    "lightning_val_offset": train_length,
    "lightning_val_length": LIGHTNING_VAL_DAYS,
    "final_test_len": final_test_len,
    "tickers": list(EQUITY_TICKERS),
    "eval_ticker": EVAL_TICKER,
    "target_suffixes": TARGET_SUFFIXES,
    "covariate_suffixes": COVARIATE_SUFFIXES,
    "shared_covariate_columns": SHARED_COVARIATE_COLUMNS,
    "eval_target_columns": eval_target_columns,
    "eval_covariate_columns": eval_covariate_columns,
    "momentum_window": MOMENTUM_WINDOW,
    "ma_window": MA_WINDOW,
}
with open(f"{OUT_DIR}/split_info_v4.json", "w") as f:
    json.dump(split_info, f, indent=2, default=str)

print(f"\nSplit info saved to {OUT_DIR}/split_info_v4.json")
