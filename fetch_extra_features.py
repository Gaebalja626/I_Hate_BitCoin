"""
추가 피처 수집: Fear & Greed Index, Funding Rate
"""
import os
import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

os.environ["SSL_CERT_FILE"] = "C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem"
os.environ["REQUESTS_CA_BUNDLE"] = "C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem"


def fetch_fear_greed(start_date="2020-01-01", end_date="2026-01-01"):
    """
    Fear & Greed Index (alternative.me API).
    일별 데이터 → 1h로 forward fill.
    """
    cache_path = "datasets/fear_greed.csv"
    if os.path.exists(cache_path):
        print(f"[캐시] {cache_path} 로드")
        df = pd.read_csv(cache_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # API: 최대 limit으로 전체 기간
    days = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days
    url = f"https://api.alternative.me/fng/?limit={days}&format=json"
    print(f"[수집] Fear & Greed Index ({days}일)")

    resp = requests.get(url, timeout=30)
    data = resp.json().get("data", [])

    rows = []
    for item in data:
        ts = int(item["timestamp"])
        dt = datetime.utcfromtimestamp(ts)
        rows.append({"date": dt.strftime("%Y-%m-%d"), "fear_greed": int(item["value"])})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    print(f"[저장] {cache_path} ({len(df)}행)")
    return df


def fetch_funding_rate(start_date="2020-01-01", end_date="2026-01-01"):
    """
    바이낸스 BTCUSDT 펀딩레이트 (8h 간격).
    """
    cache_path = "datasets/funding_rate.csv"
    if os.path.exists(cache_path):
        print(f"[캐시] {cache_path} 로드")
        df = pd.read_csv(cache_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    print(f"[수집] Funding Rate | {start_date} ~ {end_date}")
    all_data = []
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    while start_ts < end_ts:
        url = (
            f"https://fapi.binance.com/fapi/v1/fundingRate"
            f"?symbol=BTCUSDT&startTime={start_ts}&limit=1000"
        )
        try:
            resp = requests.get(url, timeout=15)
            data = resp.json()
        except Exception as e:
            print(f"  [에러] {e}")
            break

        if not data:
            break

        for item in data:
            ts = int(item["fundingTime"])
            dt = datetime.utcfromtimestamp(ts / 1000)
            all_data.append({
                "date": dt,
                "funding_rate": float(item["fundingRate"]),
            })

        start_ts = int(data[-1]["fundingTime"]) + 1
        dt_str = datetime.utcfromtimestamp(start_ts / 1000).strftime("%Y-%m-%d")
        print(f"  수집중... {dt_str}", end="\r")
        time.sleep(0.1)

    df = pd.DataFrame(all_data)
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    print(f"\n[저장] {cache_path} ({len(df)}행)")
    return df


def merge_extra_features(df_1h, fear_greed_df, funding_df):
    """
    1h OHLCV에 Fear&Greed, Funding Rate를 merge.
    - Fear&Greed: 일별 → 1h forward fill
    - Funding Rate: 8h → 1h forward fill + 누적/이동평균
    """
    df = df_1h.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Fear & Greed (일별 → 1일 shift하여 전일 값만 사용, 미래 정보 유출 방지)
    fg = fear_greed_df.copy()
    fg["date"] = pd.to_datetime(fg["date"])
    fg = fg.sort_values("date")
    fg["date"] = fg["date"] + pd.Timedelta(days=1)  # 전일 값을 다음날에 사용
    fg["date_only"] = fg["date"].dt.date
    df["date_only"] = df["date"].dt.date
    df = df.merge(fg[["date_only", "fear_greed"]], on="date_only", how="left")
    df["fear_greed"] = df["fear_greed"].ffill().bfill()
    df.drop(columns=["date_only"], inplace=True)

    # Funding Rate (8h → asof merge)
    if len(funding_df) > 0:
        fr = funding_df.copy()
        fr["date"] = pd.to_datetime(fr["date"])
        fr = fr.sort_values("date")
        df = df.sort_values("date")
        df = pd.merge_asof(df, fr[["date", "funding_rate"]], on="date", direction="backward")
        df["funding_rate"] = df["funding_rate"].ffill().fillna(0)

        # 파생 피처
        df["funding_rate_ma8"] = df["funding_rate"].rolling(8, min_periods=1).mean()
        df["funding_rate_ma24"] = df["funding_rate"].rolling(24, min_periods=1).mean()
        df["funding_rate_cumsum_24h"] = df["funding_rate"].rolling(24, min_periods=1).sum()
    else:
        df["funding_rate"] = 0
        df["funding_rate_ma8"] = 0
        df["funding_rate_ma24"] = 0
        df["funding_rate_cumsum_24h"] = 0

    print(f"[피처 병합] fear_greed + funding_rate → {len(df)}행")
    return df


if __name__ == "__main__":
    fg = fetch_fear_greed()
    print(f"Fear&Greed: {fg['date'].min()} ~ {fg['date'].max()} ({len(fg)}행)")

    fr = fetch_funding_rate()
    print(f"Funding Rate: {fr['date'].min()} ~ {fr['date'].max()} ({len(fr)}행)")
