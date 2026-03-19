"""
바이낸스 데이터 수집 모듈 (ccxt)
"""
import os
from datetime import datetime

import ccxt
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

from config import DATA_DIR, TECH_INDICATORS


def fetch_btc_data(pair="BTC/USDT", timeframe="1h",
                   start_date="2023-01-01", end_date="2025-01-01"):
    """바이낸스에서 OHLCV 데이터를 수집합니다."""
    exchange = ccxt.binance({"enableRateLimit": True})
    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    end_ts = exchange.parse8601(f"{end_date}T00:00:00Z")
    all_ohlcv = []

    print(f"[데이터 수집] {pair} {timeframe} | {start_date} ~ {end_date}")

    while since < end_ts:
        ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        dt_str = datetime.utcfromtimestamp(since / 1000).strftime("%Y-%m-%d %H:%M")
        print(f"  수집중... {dt_str}", end="\r")

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[df["timestamp"] <= end_ts].copy()
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(np.float64)

    print(f"\n[완료] 총 {len(df)}개 캔들 ({df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")
    return df


def add_technical_indicators(df, indicator_list=None):
    """stockstats 기반 보조지표 추가."""
    if indicator_list is None:
        indicator_list = TECH_INDICATORS

    temp = df[["date", "open", "high", "low", "close", "volume"]].copy()
    stock_df = Sdf.retype(temp)

    for indicator in indicator_list:
        try:
            df[indicator] = stock_df[indicator].values
        except Exception as e:
            print(f"  [경고] {indicator} 계산 실패: {e}")
            df[indicator] = 0.0

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    print(f"[보조지표] {len(indicator_list)}개 추가 → 최종 {len(df)}행")
    return df


def load_or_fetch(pair, timeframe, start_date, end_date, indicator_list=None):
    """캐시된 데이터가 있으면 로드, 없으면 수집 후 저장."""
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, f"btc_raw_{timeframe}.csv")

    if os.path.exists(csv_path):
        print(f"[캐시] {csv_path} 로드")
        return pd.read_csv(csv_path)

    raw = fetch_btc_data(pair, timeframe, start_date, end_date)
    raw = add_technical_indicators(raw, indicator_list)
    raw.to_csv(csv_path, index=False)
    print(f"[저장] {csv_path} ({len(raw)}행)")
    return raw
