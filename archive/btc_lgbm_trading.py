"""
Bitcoin ML Trading - LightGBM 부스팅 모델
바이낸스(ccxt)에서 BTC/USDT 데이터 수집 → 보조지표 + 피처엔지니어링 → LightGBM 학습 → 백테스트
"""
import os
from datetime import datetime

import ccxt
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. 바이낸스 데이터 수집 (ccxt)
# ============================================================

def fetch_btc_data(pair="BTC/USDT", timeframe="1h",
                   start_date="2023-01-01", end_date="2025-01-01"):
    """바이낸스에서 OHLCV 데이터를 가져옵니다."""
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
        print(f"  수집중... {datetime.utcfromtimestamp(since/1000).strftime('%Y-%m-%d %H:%M')}", end="\r")

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[df["timestamp"] <= end_ts].copy()
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(np.float64)

    print(f"\n[완료] 총 {len(df)}개 캔들 수집 ({df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")
    return df


# ============================================================
# 2. 보조지표 추가 (stockstats)
# ============================================================

TECH_INDICATORS = [
    "macd", "macds", "macdh",     # MACD 라인, 시그널, 히스토그램
    "boll_ub", "boll_lb",          # 볼린저밴드
    "rsi_14",                      # RSI (14)
    "cci_14",                      # CCI (14)
    "dx_14",                       # DX (14)
    "close_10_sma", "close_30_sma", "close_60_sma",  # 이동평균
    "close_10_ema", "close_30_ema",                   # 지수이동평균
    "atr_14",                      # ATR (변동성)
    "kdjk", "kdjd",                # 스토캐스틱 KDJ
]

def add_technical_indicators(df, indicator_list=None):
    """stockstats로 보조지표를 계산합니다."""
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
    print(f"[보조지표] {len(indicator_list)}개 추가 완료 → 최종 {len(df)}행")
    return df


# ============================================================
# 3. 피처 엔지니어링 (부스팅 모델용)
# ============================================================

def create_features(df, indicator_list=None, lookahead=6):
    """
    부스팅 모델용 피처를 생성합니다.
    - 수익률 (다양한 룩백)
    - 거래량 변화율
    - 가격 패턴 (캔들 바디, 윗꼬리, 아랫꼬리)
    - 보조지표 값
    - 타겟: lookahead 기간 후 상승(1) / 하락(0)
    """
    if indicator_list is None:
        indicator_list = TECH_INDICATORS

    feat = df.copy()

    # 수익률 피처
    for lag in [1, 3, 6, 12, 24]:
        feat[f"ret_{lag}"] = feat["close"].pct_change(lag)

    # 거래량 변화율
    for lag in [1, 6, 24]:
        feat[f"vol_chg_{lag}"] = feat["volume"].pct_change(lag)

    # 거래량 이동평균 대비
    feat["vol_ma_ratio"] = feat["volume"] / feat["volume"].rolling(24).mean()

    # 캔들 패턴
    feat["candle_body"] = (feat["close"] - feat["open"]) / feat["open"]
    feat["upper_shadow"] = (feat["high"] - feat[["open", "close"]].max(axis=1)) / feat["open"]
    feat["lower_shadow"] = (feat[["open", "close"]].min(axis=1) - feat["low"]) / feat["open"]
    feat["high_low_range"] = (feat["high"] - feat["low"]) / feat["low"]

    # 이동평균 대비 가격 위치
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

    # 타겟: lookahead 기간 후 수익률 기준
    feat["future_ret"] = feat["close"].shift(-lookahead) / feat["close"] - 1
    feat["target"] = (feat["future_ret"] > 0).astype(int)

    # NaN 제거
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna().reset_index(drop=True)

    # 피처 목록
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

    # 실제 존재하는 컬럼만
    feature_cols = [c for c in feature_cols if c in feat.columns]

    print(f"[피처] {len(feature_cols)}개 생성 완료")
    return feat, feature_cols


# ============================================================
# 4. LightGBM 학습
# ============================================================

def train_lightgbm(train_df, feature_cols):
    """LightGBM 모델 학습 (early stopping + validation)"""
    # 시계열이므로 뒤쪽 20%를 validation으로 사용
    val_split = int(len(train_df) * 0.8)
    X_train = train_df[feature_cols].iloc[:val_split]
    y_train = train_df["target"].iloc[:val_split]
    X_val = train_df[feature_cols].iloc[val_split:]
    y_val = train_df["target"].iloc[val_split:]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,           # 과적합 방지: 줄임
        "max_depth": 6,             # 깊이 제한
        "learning_rate": 0.02,      # 느리게 학습
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "min_child_samples": 50,    # 최소 샘플 늘림
        "lambda_l1": 0.1,           # L1 정규화
        "lambda_l2": 1.0,           # L2 정규화
        "verbose": -1,
        "n_jobs": -1,
    }

    print(f"\n{'='*60}")
    print(f"[학습 시작] LightGBM | 학습: {len(X_train)} | 검증: {len(X_val)}")
    print(f"{'='*60}")

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.log_evaluation(100),
            lgb.early_stopping(stopping_rounds=50),
        ],
    )

    # 피처 중요도 출력
    importance = model.feature_importance(importance_type="gain")
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values("importance", ascending=False)

    print(f"\n[피처 중요도 Top 10]")
    for _, row in feat_imp.head(10).iterrows():
        print(f"  {row['feature']:25s}  {row['importance']:.1f}")

    os.makedirs("trained_models", exist_ok=True)
    model.save_model("trained_models/lgbm_btc.txt")
    print(f"\n[학습 완료] 모델 저장: trained_models/lgbm_btc.txt")

    return model, feat_imp


# ============================================================
# 5. 백테스트 (시그널 기반)
# ============================================================

def backtest(model, test_df, feature_cols, threshold=0.5, trading_fee=0.001):
    """
    LightGBM 예측 기반 백테스트
    - 상승 확률 > threshold → 매수(보유)
    - 상승 확률 <= threshold → 매도(현금)
    """
    X_test = test_df[feature_cols]
    y_test = test_df["target"]
    probs = model.predict(X_test)
    preds = (probs > threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    print(f"\n{'='*60}")
    print(f"[테스트 성능]")
    print(f"  정확도: {acc:.4f}")
    print(f"  상승 예측 비율: {preds.mean():.2%}")
    print(f"{'='*60}")

    # 시뮬레이션
    balance = 100_000.0
    holdings = 0.0
    initial = balance
    total_trades = 0
    portfolio_values = [initial]

    prices = test_df["close"].values

    for i in range(len(prices) - 1):
        price = prices[i]
        signal = preds[i]

        if signal == 1 and holdings == 0:
            # 매수
            holdings = balance / price * (1 - trading_fee)
            balance = 0
            total_trades += 1
        elif signal == 0 and holdings > 0:
            # 매도
            balance = holdings * price * (1 - trading_fee)
            holdings = 0
            total_trades += 1

        total_asset = balance + holdings * prices[i + 1]
        portfolio_values.append(total_asset)

    # 마지막에 보유중이면 매도
    if holdings > 0:
        balance = holdings * prices[-1] * (1 - trading_fee)
        holdings = 0
        portfolio_values[-1] = balance

    final_asset = portfolio_values[-1]
    model_ret = (final_asset - initial) / initial * 100
    bh_ret = (prices[-1] / prices[0] - 1) * 100

    print(f"\n[백테스트 결과]")
    print(f"  초기 자산:      ${initial:,.0f}")
    print(f"  최종 자산:      ${final_asset:,.0f}")
    print(f"  LightGBM 수익률: {model_ret:+.2f}%")
    print(f"  Buy&Hold 수익률: {bh_ret:+.2f}%")
    print(f"  총 거래 횟수:    {total_trades}")
    print(f"{'='*60}")

    # 차트 저장
    os.makedirs("results", exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    # 포트폴리오 가치
    ax1 = axes[0]
    pv_norm = np.array(portfolio_values) / initial
    bh_norm = prices / prices[0]
    ax1.plot(pv_norm, label=f"LightGBM ({model_ret:+.2f}%)", linewidth=2)
    ax1.plot(bh_norm, label=f"Buy & Hold ({bh_ret:+.2f}%)", linewidth=2, alpha=0.7)
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("BTC/USDT Trading - LightGBM vs Buy & Hold")
    ax1.set_ylabel("Portfolio Value (normalized)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 예측 확률
    ax2 = axes[1]
    ax2.plot(probs, alpha=0.6, linewidth=0.5, color="steelblue")
    ax2.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold={threshold}")
    ax2.set_ylabel("상승 확률")
    ax2.set_xlabel("Step")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = f"results/btc_lgbm_backtest_t{threshold}.png"
    plt.savefig(chart_path, dpi=150)
    print(f"[차트 저장] {chart_path}")

    return portfolio_values


# ============================================================
# 6. 메인 실행
# ============================================================

if __name__ == "__main__":
    # --- 설정 ---
    PAIR = "BTC/USDT"
    TIMEFRAME = "1h"
    TRAIN_START = "2023-01-01"
    TEST_START = "2024-06-01"
    DATA_END = "2025-01-01"
    LOOKAHEAD = 6  # 6시간 후 상승/하락 예측

    DATA_DIR = "datasets"
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- 1) 데이터 수집 ---
    raw_csv = f"{DATA_DIR}/btc_raw_{TIMEFRAME}.csv"

    if os.path.exists(raw_csv):
        print("[캐시] 기존 데이터 로드")
        raw = pd.read_csv(raw_csv)
    else:
        raw = fetch_btc_data(PAIR, TIMEFRAME, TRAIN_START, DATA_END)
        raw = add_technical_indicators(raw, TECH_INDICATORS)
        raw.to_csv(raw_csv, index=False)
        print(f"[저장] {raw_csv} ({len(raw)}행)")

    # --- 2) 피처 생성 ---
    df, feature_cols = create_features(raw, TECH_INDICATORS, lookahead=LOOKAHEAD)

    # --- 3) Train/Test 분할 ---
    df["date"] = pd.to_datetime(df["date"])
    train_df = df[df["date"] < TEST_START].reset_index(drop=True)
    test_df = df[df["date"] >= TEST_START].reset_index(drop=True)
    print(f"\n학습: {len(train_df)}행 | 테스트: {len(test_df)}행")
    print(f"피처: {feature_cols}")

    # --- 4) 학습 ---
    model, feat_imp = train_lightgbm(train_df, feature_cols)

    # --- 5) 백테스트 ---
    # threshold 0.5 (기본) + 0.55 (보수적) 두 가지로 백테스트
    backtest(model, test_df, feature_cols, threshold=0.5)
    backtest(model, test_df, feature_cols, threshold=0.55)
