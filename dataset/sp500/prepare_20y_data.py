"""
Download 20 years of multivariate S&P 500 (^GSPC) OHLCV data and split it
chronologically into:

  [ TRAIN+VAL region ] [ FINAL_TEST region (~1 trading year, held out) ]

Two CSVs are written:
  - sp500_20y_trainval.csv : fed into uni2ts's data builder (never sees
    FINAL_TEST, so Lightning training/validation/early-stopping cannot leak
    into the final backtest).
  - sp500_20y_full.csv     : the complete 20y series, used only by the
    separate final-backtest script (zero-shot vs fine-tuned comparison).

Also prints the exact train_length / offset / eval_length numbers needed for
the Hydra CLI overrides in cli/conf/finetune/data/sp500.yaml and
cli/conf/finetune/val_data/sp500.yaml, since these depend on how many rows
yfinance actually returns.
"""

import json

import pandas as pd
import yfinance as yf

TICKER = "^GSPC"
YEARS = "20y"
FINAL_TEST_DAYS = 252  # ~1 trading year held out, never touched by training
LIGHTNING_VAL_DAYS = 252  # slice of the remaining data used for early stopping

OUT_DIR = "dataset/sp500"

df = yf.download(TICKER, period=YEARS, interval="1d", auto_adjust=True)
df.columns = df.columns.get_level_values(0)  # drop the ticker level
df = df[["Open", "High", "Low", "Close", "Volume"]]
df = df.asfreq("B")
df = df.ffill()

total_len = len(df)
final_test_start = total_len - FINAL_TEST_DAYS
trainval_len = final_test_start  # rows [0, trainval_len) go to the CSV uni2ts sees
train_length = trainval_len - LIGHTNING_VAL_DAYS  # what generate_finetune_builder calls train_length

full_path = f"{OUT_DIR}/sp500_20y_full.csv"
trainval_path = f"{OUT_DIR}/sp500_20y_trainval.csv"

df.to_csv(full_path)
df.iloc[:trainval_len].to_csv(trainval_path)

print(f"Total rows (20y, business days): {total_len}")
print(f"Date range: {df.index[0].date()} -> {df.index[-1].date()}")
print(f"Final test region: rows [{final_test_start}, {total_len}) "
      f"= {df.index[final_test_start].date()} -> {df.index[-1].date()}")
print(f"Train+val region written to {trainval_path}: rows [0, {trainval_len})")
print(f"  train_length (for finetune/data/sp500.yaml)      = {train_length}")
print(f"  offset       (for finetune/val_data/sp500.yaml)  = {train_length}")
print(f"  eval_length  (for finetune/val_data/sp500.yaml)  = {LIGHTNING_VAL_DAYS}")

split_info = {
    "total_len": total_len,
    "final_test_start": final_test_start,
    "trainval_len": trainval_len,
    "train_length": train_length,
    "lightning_val_offset": train_length,
    "lightning_val_length": LIGHTNING_VAL_DAYS,
    "final_test_days": FINAL_TEST_DAYS,
    "variates": ["Open", "High", "Low", "Close", "Volume"],
}
with open(f"{OUT_DIR}/split_info.json", "w") as f:
    json.dump(split_info, f, indent=2, default=str)

print(f"\nSplit info saved to {OUT_DIR}/split_info.json")
