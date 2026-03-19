"""
3-Action LightGBM 실험 (Buy / Hold / Sell)
- 미래 수익률 기반 최적 행동 라벨 → LightGBM 3-class 분류
- dead zone 스윕으로 최적 threshold 탐색
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

import config as cfg
from data_collector import load_or_fetch
from feature_engineer import create_features
from models import lgbm_model
from backtester import run_3action
from visualizer import plot_equity_curve, plot_drawdown, plot_feature_importance
from experiment import create_experiment, save_metrics
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


def print_plan(log, buy_th):
    plan = f"""
{'='*60}
  BTC/USDT 3-Action LightGBM 실험
{'='*60}

[전략]
  3-class 분류: Buy(0) / Hold(1) / Sell(2)
  라벨 기준: 미래 {cfg.LOOKAHEAD}시간 수익률
    > +{buy_th*100:.1f}% → Buy
    < -{buy_th*100:.1f}% → Sell
    그 사이   → Hold (dead zone)

[데이터]
  학습 기간:   {cfg.TRAIN_START} ~ {cfg.TEST_START}
  테스트 기간: {cfg.TEST_START} ~ {cfg.DATA_END}

[모델]
  LightGBM multiclass (softmax)
  검증 비율: {cfg.LGBM_VAL_RATIO*100:.0f}%
  Early stopping: {cfg.LGBM_EARLY_STOP}

[백테스트]
  초기 자산: ${cfg.INITIAL_BALANCE:,.0f}
  수수료:   {cfg.TRADING_FEE*100:.1f}%
  Buy → 매수, Sell → 매도, Hold → 유지
{'='*60}"""
    for line in plan.strip().split("\n"):
        log.info(line)


def plot_action_distribution(actions, test_df, save_path):
    """액션 분포 + 시계열 차트."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # 1) 파이차트
    labels = ["Buy", "Hold", "Sell"]
    counts = [(actions == i).sum() for i in range(3)]
    colors = ["#2196F3", "#757575", "#E53935"]
    axes[0].pie(counts, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 11})
    axes[0].set_title("액션 분포", fontsize=12, fontweight="bold")

    # 2) 가격 + 액션 오버레이
    prices = test_df["close"].values
    axes[1].plot(prices, color="#333", linewidth=0.8, alpha=0.7)
    buy_idx = np.where(actions == 0)[0]
    sell_idx = np.where(actions == 2)[0]
    axes[1].scatter(buy_idx, prices[buy_idx], marker="^", color="#2196F3",
                    s=8, alpha=0.5, label="Buy")
    axes[1].scatter(sell_idx, prices[sell_idx], marker="v", color="#E53935",
                    s=8, alpha=0.5, label="Sell")
    axes[1].set_title("매매 시점", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Price (USDT)")
    axes[1].legend(fontsize=9)
    sns.despine(ax=axes[1], left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [차트] {save_path}")


def plot_3action_sweep(all_results, thresholds, save_path):
    """dead zone threshold별 성과 비교."""
    returns = [r["model_return_pct"] for r in all_results]
    mdds = [r["mdd_pct"] for r in all_results]
    sharpes = [r["sharpe_ratio"] for r in all_results]
    trades = [r["total_trades"] for r in all_results]
    bh_ret = all_results[0]["buyhold_return_pct"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    th_labels = [f"{t*100:.1f}%" for t in thresholds]

    # 수익률
    ax = axes[0, 0]
    colors = ["#43A047" if r > 0 else "#E53935" for r in returns]
    ax.bar(range(len(th_labels)), returns, color=colors, edgecolor="white")
    ax.axhline(y=bh_ret, color="#FF9800", linestyle="--", label=f"B&H {bh_ret:+.1f}%")
    ax.set_xticks(range(len(th_labels)))
    ax.set_xticklabels(th_labels)
    ax.set_title("수익률 (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Dead Zone Threshold")
    ax.legend(fontsize=9)
    for i, v in enumerate(returns):
        ax.text(i, v + (1 if v >= 0 else -2), f"{v:+.1f}", ha="center", fontsize=9)
    sns.despine(ax=ax, left=True, bottom=True)

    # MDD
    ax = axes[0, 1]
    ax.bar(range(len(th_labels)), mdds, color="#E53935", alpha=0.7, edgecolor="white")
    ax.set_xticks(range(len(th_labels)))
    ax.set_xticklabels(th_labels)
    ax.set_title("MDD (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Dead Zone Threshold")
    sns.despine(ax=ax, left=True, bottom=True)

    # Sharpe
    ax = axes[1, 0]
    colors = ["#2196F3" if s > 0 else "#757575" for s in sharpes]
    ax.bar(range(len(th_labels)), sharpes, color=colors, edgecolor="white")
    ax.set_xticks(range(len(th_labels)))
    ax.set_xticklabels(th_labels)
    ax.set_title("Sharpe Ratio", fontsize=12, fontweight="bold")
    ax.set_xlabel("Dead Zone Threshold")
    sns.despine(ax=ax, left=True, bottom=True)

    # 거래 횟수
    ax = axes[1, 1]
    ax.bar(range(len(th_labels)), trades, color="#FF9800", alpha=0.7, edgecolor="white")
    ax.set_xticks(range(len(th_labels)))
    ax.set_xticklabels(th_labels)
    ax.set_title("거래 횟수", fontsize=12, fontweight="bold")
    ax.set_xlabel("Dead Zone Threshold")
    for i, v in enumerate(trades):
        ax.text(i, v + 5, str(v), ha="center", fontsize=9)
    sns.despine(ax=ax, left=True, bottom=True)

    fig.suptitle("3-Action Dead Zone Sweep", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [차트] {save_path}")


def main():
    log = setup_logger(name="lgbm_3action")

    # ── 1) 데이터 ──
    log.info("[STEP 1] 데이터 로드")
    raw = load_or_fetch(cfg.PAIR, cfg.TIMEFRAME, cfg.TRAIN_START, cfg.DATA_END, cfg.TECH_INDICATORS)

    # ── 2) Dead zone 스윕 ──
    all_results = []

    for buy_th in cfg.ACTION_THRESHOLDS:
        log.info(f"\n{'='*60}")
        log.info(f"[실험] Dead Zone Threshold = {buy_th*100:.1f}%")
        log.info(f"{'='*60}")

        print_plan(log, buy_th)

        # 피처 생성 (threshold별로 라벨이 달라짐)
        log.info("[STEP 2] 피처 엔지니어링")
        df, feature_cols = create_features(raw, cfg.TECH_INDICATORS,
                                           lookahead=cfg.LOOKAHEAD,
                                           buy_threshold=buy_th)

        # 라벨 분포
        for cls, name in [(0, "Buy"), (1, "Hold"), (2, "Sell")]:
            cnt = (df["target_3class"] == cls).sum()
            log.info(f"  {name}: {cnt} ({cnt/len(df):.1%})")

        # Train/Test
        df["date"] = pd.to_datetime(df["date"])
        train_df = df[df["date"] < cfg.TEST_START].reset_index(drop=True)
        test_df = df[df["date"] >= cfg.TEST_START].reset_index(drop=True)
        log.info(f"  학습: {len(train_df)}행 | 테스트: {len(test_df)}행")

        # 학습
        log.info("[STEP 3] LightGBM 3-class 학습")
        model, feat_imp, train_info = lgbm_model.train_3class(train_df, feature_cols)

        # 예측
        actions, probs = lgbm_model.predict_3class(model, test_df, feature_cols)
        log.info(f"  예측 액션 분포: Buy={sum(actions==0)}, Hold={sum(actions==1)}, Sell={sum(actions==2)}")

        # 백테스트
        log.info("[STEP 4] 백테스트")
        result = run_3action(actions, test_df)
        result["dead_zone"] = buy_th

        log.info(f"  수익률: {result['model_return_pct']:+.2f}% | B&H: {result['buyhold_return_pct']:+.2f}%")
        log.info(f"  MDD: {result['mdd_pct']:.2f}% | Sharpe: {result['sharpe_ratio']:.2f}")
        log.info(f"  거래: {result['total_trades']}회 | 정확도: {result['accuracy']}")

        all_results.append(result)

        # 결과 저장 (최초 threshold만 상세 차트)
        if buy_th == cfg.BUY_THRESHOLD:
            exp_config = {
                "strategy": "3-action",
                "buy_threshold": buy_th,
                "model": "LightGBM multiclass",
                "test_period": f"{cfg.TEST_START} ~ {cfg.DATA_END}",
                "train_info": train_info,
                "action_thresholds": cfg.ACTION_THRESHOLDS,
            }
            exp_dir = create_experiment("lgbm_3action", exp_config)
            add_experiment_log(log, exp_dir)

            save_metrics(exp_dir, result)
            lgbm_model.save_model(model, os.path.join(exp_dir, "model.txt"))
            plot_feature_importance(feat_imp, os.path.join(exp_dir, "feature_importance.png"))
            plot_equity_curve(result, test_df, os.path.join(exp_dir, "equity.png"))
            plot_drawdown(result, os.path.join(exp_dir, "drawdown.png"))
            plot_action_distribution(actions, test_df, os.path.join(exp_dir, "action_dist.png"))

    # 스윕 차트
    if len(all_results) > 1:
        plot_3action_sweep(all_results, cfg.ACTION_THRESHOLDS,
                           os.path.join(exp_dir, "deadzone_sweep.png"))

    # 최종 요약
    log.info(f"\n{'='*60}")
    log.info("[최종 요약] Dead Zone Sweep 결과")
    log.info(f"{'='*60}")
    log.info(f"  {'Threshold':>10} | {'수익률':>8} | {'B&H':>8} | {'MDD':>8} | {'Sharpe':>7} | {'거래':>5}")
    log.info(f"  {'-'*55}")
    for r in all_results:
        log.info(
            f"  {r['dead_zone']*100:>9.1f}% | "
            f"{r['model_return_pct']:>+7.2f}% | "
            f"{r['buyhold_return_pct']:>+7.2f}% | "
            f"{r['mdd_pct']:>7.2f}% | "
            f"{r['sharpe_ratio']:>6.2f} | "
            f"{r['total_trades']:>5}"
        )
    log.info(f"{'='*60}")
    log.info(f"[완료] 결과 폴더: {exp_dir}")


if __name__ == "__main__":
    main()
