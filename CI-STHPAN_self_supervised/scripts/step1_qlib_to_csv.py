#!/usr/bin/env python3
"""
将 my_qlib_data（Qlib A 股指数成分，如 CSI300 / CSI500）转为 CI-STHPAN 所需的逐股票 CSV。

标的来源由 INSTRUMENT_FILE 指定（instruments/ 下文件名），在日期窗口内取成分并集。
保留上交所(SH)、深交所(SZ)、北交所(BJ) 全部标的，不因交易所丢弃（Qlib 代码形如 SH600000、SZ000001、BJ430047）。

输出命名仍为 AShare_*，与训练代码里 market=AShare、pred_dataset 中的划分一致。

输出目录（与 datautils.py 中 stock 的 data_path 一致）：
  src/data/datasets/stock/
  ├── AShare_tickers_qualify_dr-0.98_min-5_smooth.csv
  ├── AShare_aver_line_dates.csv
  └── 2020-01-02/
      ├── AShare_sh600000_1.csv
      ├── AShare_sz000001_1.csv
      └── ...

Bin 格式：首个 float32 为在全局日历中的起始下标，其余为逐日数据。
"""

import os
import struct
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# ── 配置 ─────────────────────────────────────────────────────────────────
QLIB_ROOT = os.path.expanduser("~/Desktop/my_qlib_data")
OUT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "src", "data", "datasets", "stock")
)
MARKET = "AShare"
# 与 datautils.get_dls 里 stock 的 data_path 子目录名一致
DATE_FOLDER = "2020-01-02"
DATE_START = "2020-01-02"
DATE_END = "2026-03-20"
TRAIN_DAYS = 902
MIN_TRAIN_RATIO = 0.50

# Qlib instruments 文件名：csi300.txt 或 csi500.txt（放在 instruments/ 目录下）
INSTRUMENT_FILE = os.environ.get("QLIB_INSTRUMENT", "csi300.txt")


def _is_cn_exchange_ticker(ticker: str) -> bool:
    """保留 A 股三大交易所前缀（Qlib 规范：SH / SZ / BJ）。"""
    u = str(ticker).upper()
    return u.startswith("SH") or u.startswith("SZ") or u.startswith("BJ")


def _count_by_prefix(tickers):
    """统计 SH/SZ/BJ 数量（用于日志）。"""
    sh = sz = bj = 0
    for t in tickers:
        u = str(t).upper()
        if u.startswith("SH"):
            sh += 1
        elif u.startswith("SZ"):
            sz += 1
        elif u.startswith("BJ"):
            bj += 1
    return sh, sz, bj


def clean_previous_outputs(out_root: str, date_folder: str, market: str) -> None:
    """
    步骤：清理本次流水线会重写的输出，避免与旧运行结果混在一起。
    若目录/文件已存在则删除，随后由本次运行重新生成。
    """
    out = Path(out_root)
    day_dir = out / date_folder
    ticker_file = out / f"{market}_tickers_qualify_dr-0.98_min-5_smooth.csv"
    dates_file = out / f"{market}_aver_line_dates.csv"

    if day_dir.is_dir():
        shutil.rmtree(day_dir)
    if ticker_file.is_file():
        ticker_file.unlink()
    if dates_file.is_file():
        dates_file.unlink()


def read_calendar(qlib_root, date_start, date_end):
    """从 Qlib 全局日历中截取 [date_start, date_end] 的交易日序列。"""
    with open(os.path.join(qlib_root, "calendars", "day.txt")) as f:
        all_dates = [l.strip() for l in f if l.strip()]
    all_dates = np.array(all_dates)
    mask = (all_dates >= date_start) & (all_dates <= date_end)
    dates = all_dates[mask]
    return dates, all_dates


def read_qlib_instruments(qlib_root, dates, instrument_filename: str):
    """
    读取 instruments/{instrument_filename}（如 csi300.txt / csi500.txt），
    得到在日期范围内曾纳入该指数的股票代码并集（含历史上曾纳入、窗口内有效的记录）。
    """
    tickers = set()
    date_start, date_end = dates[0], dates[-1]
    path = os.path.join(qlib_root, "instruments", instrument_filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"未找到指数文件: {path}（请将 INSTRUMENT_FILE 设为存在的 instruments 文件名）"
        )
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            tk, start, end = parts[0], parts[1], parts[2]
            if start <= date_end and end >= date_start:
                tickers.add(tk)
    return sorted(tickers)


def read_bin(path, full_calendar_len):
    """读取单只股票的 qlib .day.bin；返回 (起始日历下标, 浮点序列)。"""
    with open(path, "rb") as f:
        raw = f.read()
    n = len(raw) // 4
    vals = struct.unpack(f"<{n}f", raw)
    start_idx = int(vals[0])
    data = np.array(vals[1:], dtype=np.float64)
    return start_idx, data


def align_to_calendar(start_idx, data, full_cal, sub_indices):
    """将 bin 内数据对齐到截断后的子日历；无数据处为 NaN。"""
    out = np.full(len(sub_indices), np.nan, dtype=np.float64)
    data_end_idx = start_idx + len(data)
    for i, cal_idx in enumerate(sub_indices):
        if start_idx <= cal_idx < data_end_idx:
            out[i] = data[cal_idx - start_idx]
    return out


def main():
    # 步骤 1：清空旧输出，保证本次为全量重新生成
    clean_previous_outputs(OUT_ROOT, DATE_FOLDER, MARKET)

    # 步骤 2：读交易日历，并校验区间长度（与 AShare 分支中 trade_dates 等一致）
    print("步骤 2：读取 Qlib 日历并截取区间...")
    dates, full_cal = read_calendar(QLIB_ROOT, DATE_START, DATE_END)
    assert len(dates) == 1504, f"Expected 1504 trading days, got {len(dates)}"
    print(f"  日历: {len(dates)} 个交易日 ({dates[0]} ~ {dates[-1]})")

    full_cal_list = list(full_cal)
    sub_indices = np.array([full_cal_list.index(d) for d in dates])

    # 步骤 3：读指数成分（csi300 / csi500 等），仅保留 SH/SZ/BJ A 股代码
    print(f"步骤 3：读取 instruments/{INSTRUMENT_FILE} 并保留 SH/SZ/BJ 标的...")
    tickers_all = read_qlib_instruments(QLIB_ROOT, dates, INSTRUMENT_FILE)
    tickers = [t for t in tickers_all if _is_cn_exchange_ticker(t)]
    sh0, sz0, bj0 = _count_by_prefix(tickers_all)
    sh, sz, bj = _count_by_prefix(tickers)
    print(
        f"  指数并集: {len(tickers_all)}（SH={sh0}, SZ={sz0}, BJ={bj0}）；"
        f"过滤后 A 股 SH/SZ/BJ: {len(tickers)}（SH={sh}, SZ={sz}, BJ={bj}）"
    )
    if not tickers:
        raise SystemExit("无可用标的，请检查 instruments 文件与 Qlib 数据路径。")

    # 步骤 4：创建输出子目录
    out_dir = os.path.join(OUT_ROOT, DATE_FOLDER)
    os.makedirs(out_dir, exist_ok=True)

    valid_tickers = []
    full_cal_len = len(full_cal)

    # 步骤 5：逐股票读复权收盘价、质检、特征工程并写 CSV
    print("步骤 5：逐股票导出 CSV（复权价、均线、归一化、停牌标记）...")
    for ti, ticker in enumerate(tickers):
        bin_path = os.path.join(QLIB_ROOT, "features", ticker, "adjclose.day.bin")
        if not os.path.exists(bin_path):
            print(f"  [{ti+1}/{len(tickers)}] {ticker}: adjclose.day.bin 不存在，跳过")
            continue

        start_idx, raw_data = read_bin(bin_path, full_cal_len)
        adjclose = align_to_calendar(start_idx, raw_data, full_cal, sub_indices)

        train_slice = adjclose[:TRAIN_DAYS]
        valid_count = np.sum(~np.isnan(train_slice) & (train_slice > 0))
        if valid_count < TRAIN_DAYS * MIN_TRAIN_RATIO:
            print(
                f"  [{ti+1}/{len(tickers)}] {ticker}: 训练段有效点 {valid_count}/{TRAIN_DAYS} < 50%，跳过"
            )
            continue

        s = pd.Series(adjclose)
        ma5 = s.rolling(5, min_periods=1).mean().values
        ma10 = s.rolling(10, min_periods=1).mean().values
        ma20 = s.rolling(20, min_periods=1).mean().values
        ma30 = s.rolling(30, min_periods=1).mean().values

        train_valid = train_slice[~np.isnan(train_slice) & (train_slice > 0)]
        price_max = train_valid.max()

        ma5_n = ma5 / price_max
        ma10_n = ma10 / price_max
        ma20_n = ma20 / price_max
        ma30_n = ma30 / price_max
        close_n = adjclose / price_max

        bad_mask = np.isnan(adjclose) | (adjclose <= 0)
        ma5_n[bad_mask] = -1234.0
        ma10_n[bad_mask] = -1234.0
        ma20_n[bad_mask] = -1234.0
        ma30_n[bad_mask] = -1234.0
        close_n[bad_mask] = -1234.0

        ticker_lower = ticker.lower()
        df = pd.DataFrame(
            {
                "date": dates,
                "5-day": ma5_n,
                "10-day": ma10_n,
                "20-day": ma20_n,
                "30-day": ma30_n,
                "Close": close_n,
            }
        )

        csv_name = f"{MARKET}_{ticker_lower}_1.csv"
        df.to_csv(os.path.join(out_dir, csv_name), index=False)
        valid_tickers.append(ticker_lower)

        if (ti + 1) % 50 == 0:
            print(f"  [{ti+1}/{len(tickers)}] 已处理 {ticker}")

    print(f"\n合格股票: {len(valid_tickers)} / {len(tickers)}（SH/SZ/BJ）")

    # 步骤 6：写入训练用的股票列表（文件名与 datautils 中 tickers_fname 一致）
    print("步骤 6：写入 ticker 列表文件...")
    ticker_path = os.path.join(OUT_ROOT, f"{MARKET}_tickers_qualify_dr-0.98_min-5_smooth.csv")
    with open(ticker_path, "w") as f:
        for t in valid_tickers:
            f.write(t + "\n")
    print(f"  已写入 {ticker_path}")

    # 步骤 7：写入与数据对齐的交易日历文件
    print("步骤 7：写入交易日历文件...")
    dates_path = os.path.join(OUT_ROOT, f"{MARKET}_aver_line_dates.csv")
    with open(dates_path, "w") as f:
        for d in dates:
            f.write(d + "\n")
    print(f"  已写入 {dates_path}")

    print("全部完成。")


if __name__ == "__main__":
    main()
