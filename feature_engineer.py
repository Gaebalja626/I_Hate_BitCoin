"""
피처 엔지니어링 모듈
- v1: 기본 피처 (수익률, 거래량, 캔들, 이동평균, 볼린저, 시간)
- v2: v1 + 장기 피처 (7/14/30일 수익률, SMA 크로스, 장기 변동성)
"""
import numpy as np
import pandas as pd

from config import TECH_INDICATORS


def create_features(df, indicator_list=None, lookahead=6, buy_threshold=0.005):
    """
    v1 피처 생성 (기본).
    타겟: lookahead 기간 후 상승(1) / 하락(0) + 3-class
    """
    if indicator_list is None:
        indicator_list = TECH_INDICATORS

    feat = df.copy()

    # 수익률
    for lag in [1, 3, 6, 12, 24]:
        feat[f"ret_{lag}"] = feat["close"].pct_change(lag)

    # 거래량 변화율
    for lag in [1, 6, 24]:
        feat[f"vol_chg_{lag}"] = feat["volume"].pct_change(lag)
    feat["vol_ma_ratio"] = feat["volume"] / feat["volume"].rolling(24).mean()

    # 캔들 패턴
    feat["candle_body"] = (feat["close"] - feat["open"]) / feat["open"]
    feat["upper_shadow"] = (feat["high"] - feat[["open", "close"]].max(axis=1)) / feat["open"]
    feat["lower_shadow"] = (feat[["open", "close"]].min(axis=1) - feat["low"]) / feat["open"]
    feat["high_low_range"] = (feat["high"] - feat["low"]) / feat["low"]

    # 이동평균 대비
    if "close_30_sma" in feat.columns:
        feat["price_vs_sma30"] = (feat["close"] - feat["close_30_sma"]) / feat["close_30_sma"]
    if "close_60_sma" in feat.columns:
        feat["price_vs_sma60"] = (feat["close"] - feat["close_60_sma"]) / feat["close_60_sma"]

    # 볼린저밴드 위치
    if "boll_ub" in feat.columns and "boll_lb" in feat.columns:
        boll_width = feat["boll_ub"] - feat["boll_lb"]
        feat["boll_position"] = (feat["close"] - feat["boll_lb"]) / (boll_width + 1e-8)

    # 시간 피처
    feat["date"] = pd.to_datetime(feat["date"])
    feat["hour"] = feat["date"].dt.hour
    feat["dayofweek"] = feat["date"].dt.dayofweek

    # 타겟 (이진 분류)
    feat["future_ret"] = feat["close"].shift(-lookahead) / feat["close"] - 1
    feat["target"] = (feat["future_ret"] > 0).astype(int)

    # 타겟 (3-class: Buy=0, Hold=1, Sell=2)
    feat["target_3class"] = 1  # 기본 Hold
    feat.loc[feat["future_ret"] > buy_threshold, "target_3class"] = 0   # Buy
    feat.loc[feat["future_ret"] < -buy_threshold, "target_3class"] = 2  # Sell

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # 피처 컬럼 목록 조립
    feature_cols = (
        indicator_list
        + [f"ret_{l}" for l in [1, 3, 6, 12, 24]]
        + [f"vol_chg_{l}" for l in [1, 6, 24]]
        + ["vol_ma_ratio", "candle_body", "upper_shadow", "lower_shadow", "high_low_range"]
        + ["hour", "dayofweek"]
    )
    for extra in ["price_vs_sma30", "price_vs_sma60", "boll_position"]:
        if extra in feat.columns:
            feature_cols.append(extra)
    feature_cols = [c for c in feature_cols if c in feat.columns]

    print(f"[피처] {len(feature_cols)}개 생성 완료")
    return feat, feature_cols


def create_features_v2(df, indicator_list, lookahead=24, buy_threshold=0.01):
    """
    v2 피처 생성 (장기 피처 추가).
    v1 대비 추가: 장기 수익률(48~720h), 7d/30d SMA·변동성, 14일 가격 위치 등
    """
    feat = df.copy()

    # 단기 + 장기 수익률
    for lag in [1, 3, 6, 12, 24, 48, 168, 336, 720]:
        feat[f"ret_{lag}"] = feat["close"].pct_change(lag)

    # 거래량
    for lag in [1, 6, 24]:
        feat[f"vol_chg_{lag}"] = feat["volume"].pct_change(lag)
    feat["vol_ma_ratio"] = feat["volume"] / feat["volume"].rolling(24).mean()
    feat["vol_ma_ratio_7d"] = feat["volume"] / feat["volume"].rolling(168).mean()

    # 캔들
    feat["candle_body"] = (feat["close"] - feat["open"]) / feat["open"]
    feat["upper_shadow"] = (feat["high"] - feat[["open", "close"]].max(axis=1)) / feat["open"]
    feat["lower_shadow"] = (feat[["open", "close"]].min(axis=1) - feat["low"]) / feat["open"]
    feat["high_low_range"] = (feat["high"] - feat["low"]) / feat["low"]

    # 이동평균 대비
    for col in ["close_30_sma", "close_60_sma"]:
        if col in feat.columns:
            feat[f"price_vs_{col}"] = (feat["close"] - feat[col]) / feat[col]

    # 장기 이동평균
    feat["sma_7d"] = feat["close"].rolling(168).mean()
    feat["sma_30d"] = feat["close"].rolling(720).mean()
    feat["price_vs_sma7d"] = (feat["close"] - feat["sma_7d"]) / feat["sma_7d"]
    feat["price_vs_sma30d"] = (feat["close"] - feat["sma_30d"]) / feat["sma_30d"]

    # SMA 크로스
    feat["sma_cross_7_30"] = (feat["sma_7d"] - feat["sma_30d"]) / feat["sma_30d"]

    # 장기 변동성
    feat["volatility_7d"] = feat["close"].pct_change().rolling(168).std()
    feat["volatility_30d"] = feat["close"].pct_change().rolling(720).std()

    # 고점/저점 대비 위치
    feat["high_14d"] = feat["high"].rolling(336).max()
    feat["low_14d"] = feat["low"].rolling(336).min()
    feat["price_position_14d"] = (feat["close"] - feat["low_14d"]) / (feat["high_14d"] - feat["low_14d"] + 1e-8)

    # 볼린저
    if "boll_ub" in feat.columns and "boll_lb" in feat.columns:
        bw = feat["boll_ub"] - feat["boll_lb"]
        feat["boll_position"] = (feat["close"] - feat["boll_lb"]) / (bw + 1e-8)

    # 시간
    feat["date"] = pd.to_datetime(feat["date"])
    feat["hour"] = feat["date"].dt.hour
    feat["dayofweek"] = feat["date"].dt.dayofweek

    # 타겟
    feat["future_ret"] = feat["close"].shift(-lookahead) / feat["close"] - 1
    feat["target_3class"] = 1
    feat.loc[feat["future_ret"] > buy_threshold, "target_3class"] = 0
    feat.loc[feat["future_ret"] < -buy_threshold, "target_3class"] = 2

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    feature_cols = (
        indicator_list
        + [f"ret_{l}" for l in [1, 3, 6, 12, 24, 48, 168, 336, 720]]
        + [f"vol_chg_{l}" for l in [1, 6, 24]]
        + ["vol_ma_ratio", "vol_ma_ratio_7d",
           "candle_body", "upper_shadow", "lower_shadow", "high_low_range",
           "price_vs_sma7d", "price_vs_sma30d", "sma_cross_7_30",
           "volatility_7d", "volatility_30d", "price_position_14d",
           "hour", "dayofweek"]
    )
    for extra in ["price_vs_close_30_sma", "price_vs_close_60_sma", "boll_position"]:
        if extra in feat.columns:
            feature_cols.append(extra)
    feature_cols = [c for c in feature_cols if c in feat.columns]
    print(f"[features v2] {len(feature_cols)} features")
    return feat, feature_cols
