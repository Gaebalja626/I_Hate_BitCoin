"""
Chronos 시계열 예측 기반 BTC 트레이딩 실험
- Amazon Chronos-T5-Tiny (8M params, CPU)
- 롤링 윈도우로 6시간 후 가격 예측 → 매수/매도 시그널
- 모든 출력이 콘솔 + 로그 파일에 동시 기록
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import pandas as pd

import config as cfg
from data_collector import load_or_fetch
from models import chronos_model
from experiment import create_experiment, save_metrics
from visualizer import plot_equity_curve, plot_drawdown
from logger import setup_logger, add_experiment_log

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 한글 폰트
font_path = "C:/Windows/Fonts/malgun.ttf"
if os.path.exists(font_path):
    fp = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = fp.get_name()
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", palette="muted")


CHRONOS_MODEL = "amazon/chronos-t5-tiny"
CONTEXT_LEN = 512     # 과거 512시간(~21일) 참조
HORIZON = 6           # 6시간 후 예측
NUM_SAMPLES = 20      # 확률 샘플 수
PREDICT_STEP = 1      # 매 시간 예측


def print_plan(log):
    plan = f"""
{'='*60}
  BTC/USDT Chronos 시계열 예측 실험
{'='*60}

[모델]
  Chronos:     {CHRONOS_MODEL} (8M params)
  디바이스:    CPU
  컨텍스트:    {CONTEXT_LEN}시간 (~{CONTEXT_LEN//24}일)
  예측 범위:   {HORIZON}시간 후 가격

[방식]
  롤링 윈도우로 테스트 구간 매 시간마다 예측
  예측가 > 현재가 → 매수(보유)
  예측가 <= 현재가 → 매도(현금)

[데이터]
  테스트 기간: {cfg.TEST_START} ~ {cfg.DATA_END}
  시간단위:    {cfg.TIMEFRAME}

[주의]
  CPU 추론이라 시간이 걸립니다 (~5000회 추론)
  예상 소요: 10~30분
{'='*60}"""
    for line in plan.strip().split("\n"):
        log.info(line)


def backtest_chronos(prices, signals, trading_fee=0.001):
    """Chronos 시그널 기반 백테스트."""
    balance = float(cfg.INITIAL_BALANCE)
    holdings = 0.0
    total_trades = 0
    portfolio = [cfg.INITIAL_BALANCE]

    for i in range(len(prices) - 1):
        price = prices[i]
        sig = signals[i]

        if np.isnan(sig):
            total_asset = balance + holdings * prices[i + 1]
            portfolio.append(total_asset)
            continue

        if sig == 1 and holdings == 0:
            holdings = balance / price * (1 - trading_fee)
            balance = 0
            total_trades += 1
        elif sig == 0 and holdings > 0:
            balance = holdings * price * (1 - trading_fee)
            holdings = 0
            total_trades += 1

        total_asset = balance + holdings * prices[i + 1]
        portfolio.append(total_asset)

    if holdings > 0:
        balance = holdings * prices[-1] * (1 - trading_fee)
        holdings = 0
        portfolio[-1] = balance

    final = portfolio[-1]
    initial = cfg.INITIAL_BALANCE
    model_ret = (final - initial) / initial * 100
    bh_ret = (prices[-1] / prices[0] - 1) * 100

    pv = np.array(portfolio)
    peak = np.maximum.accumulate(pv)
    mdd = ((pv - peak) / peak).min() * 100

    returns = np.diff(pv) / pv[:-1]
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(365 * 24)

    return {
        "threshold": "chronos",
        "initial_balance": initial,
        "final_asset": final,
        "model_return_pct": round(model_ret, 2),
        "buyhold_return_pct": round(bh_ret, 2),
        "total_trades": total_trades,
        "mdd_pct": round(mdd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "portfolio_values": portfolio,
        "signal_ratio": round(float(np.nanmean(signals)), 4),
        "accuracy": None,
        "predictions": [],
    }


def plot_chronos_results(test_df, predictions, pred_upper, pred_lower,
                         result, exp_dir):
    """Chronos 전용 결과 차트."""
    prices = test_df["close"].values

    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                             gridspec_kw={"height_ratios": [3, 2, 1]})

    # 1) 포트폴리오
    ax = axes[0]
    initial = result["initial_balance"]
    pv_norm = np.array(result["portfolio_values"]) / initial
    bh_norm = prices / prices[0]
    ax.plot(pv_norm, label=f"Chronos ({result['model_return_pct']:+.2f}%)",
            linewidth=2, color="#2196F3")
    ax.plot(bh_norm, label=f"Buy&Hold ({result['buyhold_return_pct']:+.2f}%)",
            linewidth=2, color="#FF9800", alpha=0.8)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Chronos-T5-Tiny: BTC/USDT Trading", fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio (normalized)")
    ax.legend(fontsize=10, frameon=True)
    sns.despine(ax=ax, left=True, bottom=True)

    # 2) 가격 + 예측
    ax = axes[1]
    ax.plot(prices, label="실제 가격", linewidth=1, color="#333", alpha=0.8)
    valid = ~np.isnan(predictions)
    idx = np.where(valid)[0]
    ax.plot(idx, predictions[valid], label="예측 (median)",
            linewidth=0.8, color="#2196F3", alpha=0.6)
    if np.any(~np.isnan(pred_upper)):
        ax.fill_between(idx, pred_lower[valid], pred_upper[valid],
                        alpha=0.15, color="#2196F3", label="80% 신뢰구간")
    ax.set_title("가격 예측 vs 실제", fontsize=12, fontweight="bold")
    ax.set_ylabel("Price (USDT)")
    ax.legend(fontsize=9, frameon=True)
    sns.despine(ax=ax, left=True, bottom=True)

    # 3) 예측 오차
    ax = axes[2]
    error = np.full(len(prices), np.nan)
    error[valid] = (predictions[valid] - prices[valid]) / prices[valid] * 100
    ax.bar(range(len(error)), error, width=1, alpha=0.5,
           color=["#43A047" if e > 0 else "#E53935" for e in np.nan_to_num(error)])
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("예측 오차 (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Error (%)")
    sns.despine(ax=ax, left=True, bottom=True)

    summary = (
        f"Return: {result['model_return_pct']:+.2f}%  |  "
        f"B&H: {result['buyhold_return_pct']:+.2f}%  |  "
        f"Sharpe: {result['sharpe_ratio']:.2f}  |  "
        f"MDD: {result['mdd_pct']:.2f}%  |  "
        f"Trades: {result['total_trades']}"
    )
    fig.text(0.5, 0.01, summary, ha="center", fontsize=11,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD", alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    save_path = os.path.join(exp_dir, "chronos_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def main():
    log = setup_logger(name="chronos_experiment")

    # ── 1) 계획 출력 ──
    print_plan(log)

    # ── 2) 데이터 로드 ──
    log.info("[STEP 1] 데이터 로드")
    raw = load_or_fetch(cfg.PAIR, cfg.TIMEFRAME, cfg.TRAIN_START, cfg.DATA_END, cfg.TECH_INDICATORS)
    raw["date"] = pd.to_datetime(raw["date"])
    test_df = raw[raw["date"] >= cfg.TEST_START].reset_index(drop=True)

    test_start_idx = raw[raw["date"] >= cfg.TEST_START].index[0]
    context_start = max(0, test_start_idx - CONTEXT_LEN)
    full_prices = raw["close"].values[context_start:]
    test_offset = test_start_idx - context_start

    log.info(f"  전체 가격 배열: {len(full_prices)}개 (컨텍스트 포함)")
    log.info(f"  테스트 시작 offset: {test_offset}")

    # ── 3) Chronos 모델 로드 ──
    log.info("[STEP 2] Chronos 모델 로드")
    pipeline = chronos_model.load_model(CHRONOS_MODEL, device="cpu")

    # ── 4) 롤링 예측 ──
    log.info("[STEP 3] 롤링 예측 (테스트 구간)")
    predictions, pred_upper, pred_lower = chronos_model.predict_rolling(
        pipeline,
        full_prices,
        context_len=CONTEXT_LEN,
        horizon=HORIZON,
        num_samples=NUM_SAMPLES,
        step=PREDICT_STEP,
    )

    test_preds = predictions[test_offset:]
    test_upper = pred_upper[test_offset:]
    test_lower = pred_lower[test_offset:]
    test_prices = full_prices[test_offset:]

    # ── 5) 시그널 생성 ──
    signals, confidence = chronos_model.generate_signals(test_prices, test_preds)
    log.info(f"  매수 시그널 비율: {np.nanmean(signals):.2%}")

    # ── 6) 백테스트 ──
    log.info("[STEP 4] 백테스트")
    result = backtest_chronos(test_prices, signals.astype(float))

    log.info(f"{'='*60}")
    log.info(f"[백테스트 결과] Chronos-T5-Tiny")
    log.info(f"  초기 자산:      ${result['initial_balance']:,.0f}")
    log.info(f"  최종 자산:      ${result['final_asset']:,.0f}")
    log.info(f"  Chronos 수익률: {result['model_return_pct']:+.2f}%")
    log.info(f"  Buy&Hold:       {result['buyhold_return_pct']:+.2f}%")
    log.info(f"  MDD:            {result['mdd_pct']:.2f}%")
    log.info(f"  Sharpe:         {result['sharpe_ratio']:.2f}")
    log.info(f"  총 거래:        {result['total_trades']}회")
    log.info(f"{'='*60}")

    # ── 7) 결과 저장 ──
    log.info("[STEP 5] 결과 저장")
    exp_config = {
        "model": CHRONOS_MODEL,
        "context_len": CONTEXT_LEN,
        "horizon": HORIZON,
        "num_samples": NUM_SAMPLES,
        "test_period": f"{cfg.TEST_START} ~ {cfg.DATA_END}",
    }
    exp_dir = create_experiment("chronos_tiny", exp_config)
    add_experiment_log(log, exp_dir)

    save_metrics(exp_dir, result)
    chart_path = plot_chronos_results(test_df, test_preds, test_upper, test_lower, result, exp_dir)
    log.info(f"  [차트] {chart_path}")
    plot_equity_curve(result, test_df, os.path.join(exp_dir, "equity.png"))
    plot_drawdown(result, os.path.join(exp_dir, "drawdown.png"))

    log.info(f"[완료] 결과 폴더: {exp_dir}")
    log.info(f"[로그] {exp_dir}/experiment.log")
    log.info(f"[전역 로그] results/all_experiments.log")


if __name__ == "__main__":
    main()
