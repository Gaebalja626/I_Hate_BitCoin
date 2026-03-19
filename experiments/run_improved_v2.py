"""
개선된 3-Action LightGBM (v2)
- Lookahead: 24h (기존 6h)
- 장기 피처 추가 (7/14/30일 수익률, SMA 크로스, 장기 변동성)
- 확률 기반 진입 (Buy/Sell 확률 > 40%)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

os.environ['SSL_CERT_FILE'] = 'C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem'
os.environ['REQUESTS_CA_BUNDLE'] = 'C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem'

import config as cfg
from data_collector import load_or_fetch, fetch_btc_data, add_technical_indicators
from models import lgbm_model
from backtester import run_3action

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import seaborn as sns

font_path = "C:/Windows/Fonts/malgun.ttf"
if os.path.exists(font_path):
    fp = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = fp.get_name()
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", palette="muted")


def create_features_v2(df, indicator_list, lookahead=24, buy_threshold=0.01):
    feat = df.copy()

    # 단기 수익률
    for lag in [1, 3, 6, 12, 24]:
        feat[f"ret_{lag}"] = feat["close"].pct_change(lag)

    # 장기 수익률
    for lag in [48, 168, 336, 720]:
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


def main():
    # ── 1) 데이터 ──
    print("[STEP 1] Data")
    train_raw = load_or_fetch(cfg.PAIR, cfg.TIMEFRAME, cfg.TRAIN_START, cfg.DATA_END, cfg.TECH_INDICATORS)
    test_raw = fetch_btc_data(cfg.PAIR, cfg.TIMEFRAME, "2025-01-01", "2026-01-01")
    test_raw = add_technical_indicators(test_raw, cfg.TECH_INDICATORS)
    full_raw = pd.concat([train_raw, test_raw]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # ── 2) 피처 ──
    LOOKAHEAD = 24
    BUY_TH = 0.01
    print(f"[STEP 2] Features v2 (lookahead={LOOKAHEAD}h, dead_zone={BUY_TH*100}%)")
    df, feature_cols = create_features_v2(full_raw, cfg.TECH_INDICATORS, lookahead=LOOKAHEAD, buy_threshold=BUY_TH)
    df["date"] = pd.to_datetime(df["date"])

    train_df = df[df["date"] < "2025-01-01"].reset_index(drop=True)
    test_df = df[df["date"] >= "2025-01-01"].reset_index(drop=True)
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    for cls, name in [(0, "Buy"), (1, "Hold"), (2, "Sell")]:
        tr = (train_df["target_3class"] == cls).sum()
        te = (test_df["target_3class"] == cls).sum()
        print(f"  {name}: train={tr}({tr/len(train_df):.1%}) test={te}({te/len(test_df):.1%})")

    # ── 3) 학습 ──
    print("[STEP 3] Train")
    model, feat_imp, train_info = lgbm_model.train_3class(train_df, feature_cols)

    # ── 4) 확률 기반 예측 ──
    print("[STEP 4] Predict (probability-based entry)")
    _, probs = lgbm_model.predict_3class(model, test_df, feature_cols)

    BUY_PROB_TH = 0.40
    SELL_PROB_TH = 0.40
    actions = np.ones(len(probs), dtype=int)  # default Hold
    actions[probs[:, 0] > BUY_PROB_TH] = 0   # Buy
    actions[probs[:, 2] > SELL_PROB_TH] = 2  # Sell
    print(f"  Buy={sum(actions==0)}, Hold={sum(actions==1)}, Sell={sum(actions==2)}")

    # ── 5) 백테스트 ──
    print("[STEP 5] Backtest")
    result = run_3action(actions, test_df)
    trade_log = result["trade_log"]

    print(f"\n{'='*60}")
    print(f"[Improved OOS] 2025.01~2026.01")
    print(f"  Changes: lookahead=24h, dead_zone=1%, long-term features, prob entry(40%)")
    print(f"{'='*60}")
    print(f"  Initial:   ${100000:,.0f}")
    print(f"  Final:     ${result['final_asset']:,.0f}")
    print(f"  Return:    {result['model_return_pct']:+.2f}%")
    print(f"  B&H:       {result['buyhold_return_pct']:+.2f}%")
    print(f"  MDD:       {result['mdd_pct']:.2f}%")
    print(f"  Sharpe:    {result['sharpe_ratio']:.2f}")
    print(f"  Trades:    {result['total_trades']}")
    print(f"  Accuracy:  {result['accuracy']}")
    print(f"{'='*60}")

    print("\n[Feature Importance Top 15]")
    for _, row in feat_imp.head(15).iterrows():
        print(f"  {row['feature']:25s} {row['importance']:.1f}")

    # ── 6) 차트 ──
    dates = test_df["date"].values
    prices = test_df["close"].values

    fig, axes = plt.subplots(3, 1, figsize=(18, 14),
                             gridspec_kw={"height_ratios": [4, 1.5, 1]})

    ax = axes[0]
    ax.plot(dates, prices, color="#333", linewidth=1, alpha=0.8, label="BTC/USDT")
    buy_times = [dates[t[0]] for t in trade_log if t[1] == "BUY"]
    buy_prices = [t[2] for t in trade_log if t[1] == "BUY"]
    sell_times = [dates[t[0]] for t in trade_log if t[1] == "SELL"]
    sell_prices = [t[2] for t in trade_log if t[1] == "SELL"]
    ax.scatter(buy_times, buy_prices, marker="^", color="#2196F3", s=100,
               zorder=5, label=f"Buy ({len(buy_times)})", edgecolors="white", linewidths=0.5)
    ax.scatter(sell_times, sell_prices, marker="v", color="#E53935", s=100,
               zorder=5, label=f"Sell ({len(sell_times)})", edgecolors="white", linewidths=0.5)
    for i in range(min(len(buy_times), len(sell_times))):
        ax.axvspan(buy_times[i], sell_times[i], alpha=0.06, color="#2196F3")
    ax.set_title(
        f"[Improved OOS] BTC/USDT 3-Action (24h lookahead)  |  "
        f"Return: {result['model_return_pct']:+.2f}%  vs  B&H: {result['buyhold_return_pct']:+.2f}%",
        fontsize=14, fontweight="bold")
    ax.set_ylabel("Price (USDT)", fontsize=11)
    ax.legend(fontsize=10, loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    sns.despine(ax=ax, left=True, bottom=True)

    ax = axes[1]
    pv = np.array(result["portfolio_values"])
    pv_norm = pv / pv[0]
    bh_norm = prices / prices[0]
    plot_dates = dates[:len(pv_norm)]
    ax.plot(plot_dates, pv_norm, color="#2196F3", linewidth=2,
            label=f"3-Action v2 ({result['model_return_pct']:+.2f}%)")
    ax.plot(dates[:len(bh_norm)], bh_norm, color="#FF9800", linewidth=2, alpha=0.7,
            label=f"Buy&Hold ({result['buyhold_return_pct']:+.2f}%)")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(plot_dates, pv_norm, 1, alpha=0.08, color="#2196F3")
    ax.set_ylabel("Portfolio", fontsize=11)
    ax.legend(fontsize=9, frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    sns.despine(ax=ax, left=True, bottom=True)

    ax = axes[2]
    peak = np.maximum.accumulate(pv)
    dd = (pv - peak) / peak * 100
    ax.fill_between(plot_dates, dd, 0, alpha=0.4, color="#E53935")
    ax.set_title(f"Drawdown (MDD: {result['mdd_pct']:.2f}%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("%", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    sns.despine(ax=ax, left=True, bottom=True)

    summary = (
        f"OOS: 2025.01~2026.01  |  Lookahead: 24h  |  Dead Zone: 1%  |  "
        f"Return: {result['model_return_pct']:+.2f}%  |  B&H: {result['buyhold_return_pct']:+.2f}%  |  "
        f"Sharpe: {result['sharpe_ratio']:.2f}  |  MDD: {result['mdd_pct']:.2f}%  |  "
        f"Trades: {result['total_trades']}"
    )
    fig.text(0.5, 0.01, summary, ha="center", fontsize=11,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD", alpha=0.8))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    save_path = "results/lgbm_3action_20260319_134505/oos_improved_v2.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Chart] {save_path}")

    # 매매 로그
    print(f"\n[Trade Log]")
    for idx, (step, action, price) in enumerate(trade_log):
        date = test_df["date"].iloc[step]
        print(f"  {idx+1:3d}. {date.strftime('%Y-%m-%d %H:%M')} | {action:4s} | ${price:,.2f}")


if __name__ == "__main__":
    main()
