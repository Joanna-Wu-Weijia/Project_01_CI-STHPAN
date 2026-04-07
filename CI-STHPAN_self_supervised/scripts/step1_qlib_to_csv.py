#!/usr/bin/env python3
"""
Convert my_qlib_data (A-Share CSI300) into per-stock CSVs
that match CI-STHPAN's expected format (NASDAQ_AAPL_1.csv style).

Output:
  src/data/datasets/stock/
  ├── AShare_tickers_qualify_dr-0.98_min-5_smooth.csv
  ├── AShare_aver_line_dates.csv
  └── 2020-01-02/
      ├── AShare_sh600000_1.csv
      └── ...

Bin format: first float32 = start_index in calendar, rest = data values.
"""

import os
import struct
import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
QLIB_ROOT   = os.path.expanduser("~/Desktop/my_qlib_data")
OUT_ROOT    = os.path.join(os.path.dirname(__file__),
                           "..", "src", "data", "datasets", "stock")
MARKET      = "AShare"
DATE_START  = "2020-01-02"
DATE_END    = "2026-03-20"
TRAIN_DAYS  = 902          
MIN_TRAIN_RATIO = 0.50    


def read_calendar(qlib_root, date_start, date_end):
    """Read calendar and filter to [date_start, date_end]."""
    with open(os.path.join(qlib_root, "calendars", "day.txt")) as f:
        all_dates = [l.strip() for l in f if l.strip()]
    all_dates = np.array(all_dates)
    mask = (all_dates >= date_start) & (all_dates <= date_end)
    dates = all_dates[mask]
    return dates, all_dates


def read_csi300_tickers(qlib_root, dates):
    """Read instruments/csi300.txt; return tickers present in date range."""
    tickers = set()
    date_start, date_end = dates[0], dates[-1]
    with open(os.path.join(qlib_root, "instruments", "csi300.txt")) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            tk, start, end = parts[0], parts[1], parts[2]
            if start <= date_end and end >= date_start:
                tickers.add(tk)
    return sorted(tickers)


def read_bin(path, full_calendar_len):
    """Read a qlib .day.bin file. Returns (start_idx, values)."""
    with open(path, "rb") as f:
        raw = f.read()
    n = len(raw) // 4
    vals = struct.unpack(f"<{n}f", raw)
    start_idx = int(vals[0])
    data = np.array(vals[1:], dtype=np.float64)
    return start_idx, data


def align_to_calendar(start_idx, data, full_cal, sub_indices):
    """Align bin data to the sub-calendar (sub_indices into full_cal).
    Returns array of length len(sub_indices), NaN where no data."""
    out = np.full(len(sub_indices), np.nan, dtype=np.float64)
    data_end_idx = start_idx + len(data)  # exclusive
    for i, cal_idx in enumerate(sub_indices):
        if start_idx <= cal_idx < data_end_idx:
            out[i] = data[cal_idx - start_idx]
    return out


def main():
    print("Reading calendar...")
    dates, full_cal = read_calendar(QLIB_ROOT, DATE_START, DATE_END)
    assert len(dates) == 1504, f"Expected 1504 trading days, got {len(dates)}"
    print(f"Calendar: {len(dates)} days ({dates[0]} ~ {dates[-1]})")

    # Map sub-calendar dates to indices in full calendar
    full_cal_list = list(full_cal)
    sub_indices = np.array([full_cal_list.index(d) for d in dates])

    print("Reading CSI300 tickers...")
    tickers = read_csi300_tickers(QLIB_ROOT, dates)
    print(f"CSI300 union tickers: {len(tickers)}")

    out_dir = os.path.join(OUT_ROOT, "2020-01-02")
    os.makedirs(out_dir, exist_ok=True)

    valid_tickers = []
    full_cal_len = len(full_cal)

    for ti, ticker in enumerate(tickers):
        bin_path = os.path.join(QLIB_ROOT, "features", ticker, "adjclose.day.bin")
        if not os.path.exists(bin_path):
            print(f"  [{ti+1}/{len(tickers)}] {ticker}: adjclose.day.bin not found, skip")
            continue

        # Read adjclose
        start_idx, raw_data = read_bin(bin_path, full_cal_len)
        adjclose = align_to_calendar(start_idx, raw_data, full_cal, sub_indices)

        # Check train coverage
        train_slice = adjclose[:TRAIN_DAYS]
        valid_count = np.sum(~np.isnan(train_slice) & (train_slice > 0))
        if valid_count < TRAIN_DAYS * MIN_TRAIN_RATIO:
            print(f"  [{ti+1}/{len(tickers)}] {ticker}: train coverage {valid_count}/{TRAIN_DAYS} < 50%, skip")
            continue

        # Compute rolling means
        s = pd.Series(adjclose)
        ma5  = s.rolling(5,  min_periods=1).mean().values
        ma10 = s.rolling(10, min_periods=1).mean().values
        ma20 = s.rolling(20, min_periods=1).mean().values
        ma30 = s.rolling(30, min_periods=1).mean().values

        # price_max = max adjclose in train set (ignoring NaN and <=0)
        train_valid = train_slice[~np.isnan(train_slice) & (train_slice > 0)]
        price_max = train_valid.max()

        # Normalize
        ma5_n  = ma5  / price_max
        ma10_n = ma10 / price_max
        ma20_n = ma20 / price_max
        ma30_n = ma30 / price_max
        close_n = adjclose / price_max

        # Mark suspension/missing: adjclose NaN or <=0 → fill -1234
        bad_mask = np.isnan(adjclose) | (adjclose <= 0)
        ma5_n[bad_mask]  = -1234.0
        ma10_n[bad_mask] = -1234.0
        ma20_n[bad_mask] = -1234.0
        ma30_n[bad_mask] = -1234.0
        close_n[bad_mask] = -1234.0

        # Build DataFrame
        # Ticker naming: SH600000 → sh600000, SZ000001 → sz000001
        ticker_lower = ticker.lower()
        df = pd.DataFrame({
            "date": dates,
            "5-day": ma5_n,
            "10-day": ma10_n,
            "20-day": ma20_n,
            "30-day": ma30_n,
            "Close": close_n,
        })

        csv_name = f"{MARKET}_{ticker_lower}_1.csv"
        df.to_csv(os.path.join(out_dir, csv_name), index=False)
        valid_tickers.append(ticker_lower)

        if (ti + 1) % 50 == 0:
            print(f"  [{ti+1}/{len(tickers)}] processed {ticker}")

    print(f"\nValid tickers: {len(valid_tickers)} / {len(tickers)}")

    # Write ticker list
    ticker_path = os.path.join(OUT_ROOT,
                               f"{MARKET}_tickers_qualify_dr-0.98_min-5_smooth.csv")
    with open(ticker_path, "w") as f:
        for t in valid_tickers:
            f.write(t + "\n")
    print(f"Wrote {ticker_path}")

    # Write dates file
    dates_path = os.path.join(OUT_ROOT, f"{MARKET}_aver_line_dates.csv")
    with open(dates_path, "w") as f:
        for d in dates:
            f.write(d + "\n")
    print(f"Wrote {dates_path}")

    print("Done!")


if __name__ == "__main__":
    main()
