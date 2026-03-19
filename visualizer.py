"""
시각화 모듈 (seaborn + matplotlib)
- 수익 곡선, 피처 중요도, 예측 분포, 종합 대시보드, 실험 비교
"""
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns


def _setup_style():
    """seaborn 스타일 + 한글 폰트 설정."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
    font_path = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False


_setup_style()

# 통일된 색상 팔레트
COLORS = {
    "model": "#2196F3",    # 파랑
    "buyhold": "#FF9800",  # 주황
    "drawdown": "#E53935", # 빨강
    "accent": "#43A047",   # 초록
    "neutral": "#757575",  # 회색
}


def plot_equity_curve(result, test_df, save_path):
    """수익 곡선 차트 (모델 vs Buy&Hold)."""
    initial = result["initial_balance"]
    pv_norm = np.array(result["portfolio_values"]) / initial
    prices = test_df["close"].values
    bh_norm = prices / prices[0]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(pv_norm, label=f"LightGBM ({result['model_return_pct']:+.2f}%)",
            linewidth=2, color=COLORS["model"])
    ax.plot(bh_norm, label=f"Buy & Hold ({result['buyhold_return_pct']:+.2f}%)",
            linewidth=2, color=COLORS["buyhold"], alpha=0.8)
    ax.axhline(y=1.0, color=COLORS["neutral"], linestyle="--", alpha=0.5)
    ax.fill_between(range(len(pv_norm)), pv_norm, 1, alpha=0.08, color=COLORS["model"])
    ax.set_title(f"BTC/USDT Equity Curve  (threshold={result['threshold']})", fontsize=13)
    ax.set_xlabel("Step")
    ax.set_ylabel("Portfolio (normalized)")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [차트] {save_path}")


def plot_drawdown(result, save_path):
    """낙폭(Drawdown) 차트."""
    pv = np.array(result["portfolio_values"])
    peak = np.maximum.accumulate(pv)
    drawdown = (pv - peak) / peak * 100

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.fill_between(range(len(drawdown)), drawdown, 0,
                    alpha=0.35, color=COLORS["drawdown"])
    ax.plot(drawdown, color=COLORS["drawdown"], linewidth=0.8)
    ax.set_title(f"Drawdown  (MDD: {result['mdd_pct']:.2f}%)", fontsize=12)
    ax.set_xlabel("Step")
    ax.set_ylabel("Drawdown (%)")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [차트] {save_path}")


def plot_feature_importance(feat_imp, save_path, top_n=20):
    """피처 중요도 수평 바 차트 (seaborn barplot)."""
    top = feat_imp.head(top_n).iloc[::-1].copy()

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    sns.barplot(data=top, x="importance", y="feature", ax=ax,
                palette="Blues_d", edgecolor="white")
    ax.set_title(f"Feature Importance (Top {top_n}, Gain)", fontsize=13)
    ax.set_xlabel("Gain")
    ax.set_ylabel("")
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [차트] {save_path}")


def plot_prediction_dist(predictions, save_path, threshold=0.5):
    """예측 확률 분포 + 시계열."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # 히스토그램 (seaborn)
    sns.histplot(predictions, bins=50, ax=axes[0], color=COLORS["model"],
                 edgecolor="white", alpha=0.7, kde=True)
    axes[0].axvline(x=threshold, color=COLORS["drawdown"], linestyle="--",
                    linewidth=2, label=f"threshold={threshold}")
    axes[0].set_title("예측 확률 분포", fontsize=12)
    axes[0].set_xlabel("P(상승)")
    axes[0].set_ylabel("빈도")
    axes[0].legend()

    # 시계열
    axes[1].plot(predictions, alpha=0.5, linewidth=0.5, color=COLORS["model"])
    axes[1].axhline(y=threshold, color=COLORS["drawdown"], linestyle="--", linewidth=1.5)
    axes[1].set_title("예측 확률 추이", fontsize=12)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("P(상승)")

    for ax in axes:
        sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [차트] {save_path}")


def plot_summary_dashboard(result, test_df, feat_imp, save_path):
    """종합 대시보드 (4-panel)."""
    initial = result["initial_balance"]
    pv = np.array(result["portfolio_values"])
    pv_norm = pv / initial
    prices = test_df["close"].values
    bh_norm = prices / prices[0]
    predictions = np.array(result["predictions"])

    peak = np.maximum.accumulate(pv)
    drawdown = (pv - peak) / peak * 100

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 2, 0.3], hspace=0.35, wspace=0.3)

    # ── 1) 수익 곡선 ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(pv_norm, label=f"Model ({result['model_return_pct']:+.2f}%)",
             linewidth=2, color=COLORS["model"])
    ax1.plot(bh_norm, label=f"B&H ({result['buyhold_return_pct']:+.2f}%)",
             linewidth=2, color=COLORS["buyhold"], alpha=0.8)
    ax1.axhline(y=1.0, color=COLORS["neutral"], linestyle="--", alpha=0.5)
    ax1.fill_between(range(len(pv_norm)), pv_norm, 1, alpha=0.06, color=COLORS["model"])
    ax1.set_title("Equity Curve", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, frameon=True, fancybox=True)
    sns.despine(ax=ax1, left=True, bottom=True)

    # ── 2) Drawdown ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(range(len(drawdown)), drawdown, 0,
                     alpha=0.35, color=COLORS["drawdown"])
    ax2.plot(drawdown, color=COLORS["drawdown"], linewidth=0.6)
    ax2.set_title(f"Drawdown (MDD: {result['mdd_pct']:.2f}%)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("%")
    sns.despine(ax=ax2, left=True, bottom=True)

    # ── 3) 피처 중요도 Top 10 ──
    ax3 = fig.add_subplot(gs[1, 0])
    top10 = feat_imp.head(10).iloc[::-1].copy()
    sns.barplot(data=top10, x="importance", y="feature", ax=ax3,
                palette="Blues_d", edgecolor="white")
    ax3.set_title("Feature Importance (Top 10)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Gain")
    ax3.set_ylabel("")
    sns.despine(ax=ax3, left=True, bottom=True)

    # ── 4) 예측 확률 분포 ──
    ax4 = fig.add_subplot(gs[1, 1])
    sns.histplot(predictions, bins=50, ax=ax4, color=COLORS["model"],
                 edgecolor="white", alpha=0.7, kde=True)
    ax4.axvline(x=result["threshold"], color=COLORS["drawdown"], linestyle="--",
                linewidth=2, label=f"threshold={result['threshold']}")
    ax4.set_title("예측 확률 분포", fontsize=12, fontweight="bold")
    ax4.set_xlabel("P(상승)")
    ax4.legend(fontsize=9)
    sns.despine(ax=ax4, left=True, bottom=True)

    # ── 하단 요약 텍스트 ──
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis("off")
    summary = (
        f"Return: {result['model_return_pct']:+.2f}%   |   "
        f"Buy&Hold: {result['buyhold_return_pct']:+.2f}%   |   "
        f"Sharpe: {result['sharpe_ratio']:.2f}   |   "
        f"MDD: {result['mdd_pct']:.2f}%   |   "
        f"Accuracy: {result['accuracy']}   |   "
        f"Trades: {result['total_trades']}"
    )
    ax_text.text(0.5, 0.5, summary, ha="center", va="center", fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", alpha=0.8))

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [차트] {save_path}")


def plot_threshold_sweep(all_results, save_path):
    """threshold별 주요 지표를 한눈에 비교하는 스윕 차트."""
    thresholds = [r["threshold"] for r in all_results]
    returns = [r["model_return_pct"] for r in all_results]
    mdds = [r["mdd_pct"] for r in all_results]
    sharpes = [r["sharpe_ratio"] for r in all_results]
    accs = [r["accuracy"] for r in all_results]
    trades = [r["total_trades"] for r in all_results]
    signals = [r["signal_ratio"] * 100 for r in all_results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # 1) 수익률
    ax = axes[0, 0]
    bars = ax.bar(range(len(thresholds)), returns, color=[
        COLORS["accent"] if r > 0 else COLORS["drawdown"] for r in returns
    ], edgecolor="white")
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)
    ax.set_title("수익률 (%)", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color=COLORS["neutral"], linestyle="--", alpha=0.5)
    ax.axhline(y=all_results[0]["buyhold_return_pct"], color=COLORS["buyhold"],
               linestyle="--", alpha=0.7, label=f"B&H {all_results[0]['buyhold_return_pct']:+.1f}%")
    ax.legend(fontsize=8)
    for i, v in enumerate(returns):
        ax.text(i, v + (1 if v >= 0 else -2), f"{v:+.1f}", ha="center", fontsize=8)
    sns.despine(ax=ax, left=True, bottom=True)

    # 2) MDD
    ax = axes[0, 1]
    ax.bar(range(len(thresholds)), mdds, color=COLORS["drawdown"],
           alpha=0.7, edgecolor="white")
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)
    ax.set_title("MDD (%)", fontsize=12, fontweight="bold")
    for i, v in enumerate(mdds):
        ax.text(i, v - 1, f"{v:.1f}", ha="center", fontsize=8)
    sns.despine(ax=ax, left=True, bottom=True)

    # 3) Sharpe
    ax = axes[0, 2]
    ax.bar(range(len(thresholds)), sharpes, color=[
        COLORS["model"] if s > 0 else COLORS["neutral"] for s in sharpes
    ], edgecolor="white")
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)
    ax.set_title("Sharpe Ratio", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color=COLORS["neutral"], linestyle="--", alpha=0.5)
    for i, v in enumerate(sharpes):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    sns.despine(ax=ax, left=True, bottom=True)

    # 4) 정확도
    ax = axes[1, 0]
    ax.bar(range(len(thresholds)), accs, color=COLORS["model"],
           alpha=0.7, edgecolor="white")
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)
    ax.set_title("정확도", fontsize=12, fontweight="bold")
    ax.axhline(y=0.5, color=COLORS["drawdown"], linestyle="--", alpha=0.5, label="50%")
    ax.legend(fontsize=8)
    for i, v in enumerate(accs):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
    sns.despine(ax=ax, left=True, bottom=True)

    # 5) 거래 횟수
    ax = axes[1, 1]
    ax.bar(range(len(thresholds)), trades, color=COLORS["buyhold"],
           alpha=0.7, edgecolor="white")
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)
    ax.set_title("거래 횟수", fontsize=12, fontweight="bold")
    for i, v in enumerate(trades):
        ax.text(i, v + 10, str(v), ha="center", fontsize=8)
    sns.despine(ax=ax, left=True, bottom=True)

    # 6) 매수 시그널 비율
    ax = axes[1, 2]
    ax.bar(range(len(thresholds)), signals, color=COLORS["accent"],
           alpha=0.7, edgecolor="white")
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)
    ax.set_title("매수 시그널 비율 (%)", fontsize=12, fontweight="bold")
    for i, v in enumerate(signals):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    sns.despine(ax=ax, left=True, bottom=True)

    fig.suptitle("Threshold Sweep Analysis", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [차트] {save_path}")


def compare_experiments(results_dirs, save_path="results/comparison.png"):
    """여러 실험의 수익 곡선을 비교합니다."""
    palette = sns.color_palette("husl", len(results_dirs))
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, exp_dir in enumerate(results_dirs):
        metrics_path = os.path.join(exp_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        pv = metrics.get("portfolio_values", [])
        if not pv:
            continue
        pv_norm = np.array(pv) / pv[0]
        label = os.path.basename(exp_dir)
        ret = metrics.get("model_return_pct", 0)
        ax.plot(pv_norm, label=f"{label} ({ret:+.2f}%)",
                linewidth=1.5, color=palette[i])

    ax.axhline(y=1.0, color=COLORS["neutral"], linestyle="--", alpha=0.5)
    ax.set_title("Experiment Comparison", fontsize=13, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Portfolio (normalized)")
    ax.legend(fontsize=8, frameon=True, fancybox=True)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[비교 차트] {save_path}")
