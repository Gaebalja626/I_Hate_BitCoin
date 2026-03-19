"""
XGBoost 투자 행동 정밀 분석
- 실제로 어떤 시장 상황에서 어떤 행동을 하는가?
- 기존 주장 검증: 변동성↑=고빈도, 변동성↓=B&H, 하락=현금보유
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

os.environ["SSL_CERT_FILE"] = "C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem"
os.environ["REQUESTS_CA_BUNDLE"] = "C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem"

import xgboost as xgb
import config as cfg
from data_collector import load_or_fetch
from backtester import run_3action
from feature_engineer import create_features_v2
from fetch_extra_features import fetch_fear_greed, fetch_funding_rate, merge_extra_features
from experiment import create_experiment

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

LA = 24; DZ = 0.01; RETRAIN_EVERY = 720; VAL_RATIO = 0.2

# ── 데이터 ──
raw = load_or_fetch(cfg.PAIR, cfg.TIMEFRAME, cfg.TRAIN_START, "2026-01-01", cfg.TECH_INDICATORS)
fg_df = fetch_fear_greed(); fr_df = fetch_funding_rate()
raw = merge_extra_features(raw, fg_df, fr_df)
df_all, fcols = create_features_v2(raw, cfg.TECH_INDICATORS, lookahead=LA, buy_threshold=DZ)
df_all["date"] = pd.to_datetime(df_all["date"])
extra_cols = ["fear_greed", "funding_rate", "funding_rate_ma8", "funding_rate_ma24", "funding_rate_cumsum_24h"]
for col in extra_cols:
    if col in df_all.columns: fcols.append(col)
fcols = [c for c in fcols if c in df_all.columns]

exp_dir = create_experiment("xgb_behavior", {"la": LA, "dz": DZ})

periods = [
    ("2024-01-01", "2025-01-01", "2024~2025"),
    ("2025-01-01", "2026-01-01", "2025~2026"),
]

for oos_start, oos_end, label in periods:
    print(f"\n{'='*70}")
    print(f"  {label} 행동 분석")
    print(f"{'='*70}")

    oos_mask = (df_all["date"] >= pd.Timestamp(oos_start)) & (df_all["date"] < pd.Timestamp(oos_end))
    oos_indices = df_all[oos_mask].index.tolist()
    oos_df = df_all.loc[oos_indices].reset_index(drop=True)

    # TTA 학습 + 예측
    all_actions = []
    all_probs = []
    i = 0
    while i < len(oos_indices):
        chunk_end = min(i + RETRAIN_EVERY, len(oos_indices))
        chunk_indices = oos_indices[i:chunk_end]
        current_time = df_all.loc[chunk_indices[0], "date"]
        train_cutoff = current_time - pd.Timedelta(hours=LA)
        train_df = df_all[df_all["date"] <= train_cutoff].reset_index(drop=True)
        pred_df = df_all.loc[chunk_indices].reset_index(drop=True)

        val_split = int(len(train_df) * (1 - VAL_RATIO))
        params = {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
                  "max_depth": 6, "learning_rate": 0.02, "subsample": 0.7,
                  "colsample_bytree": 0.7, "min_child_weight": 50,
                  "reg_alpha": 0.1, "reg_lambda": 1.0, "tree_method": "hist", "verbosity": 0}
        dtrain = xgb.DMatrix(train_df[fcols].iloc[:val_split], label=train_df["target_3class"].iloc[:val_split])
        dval = xgb.DMatrix(train_df[fcols].iloc[val_split:], label=train_df["target_3class"].iloc[val_split:])
        dpred = xgb.DMatrix(pred_df[fcols])
        model = xgb.train(params, dtrain, num_boost_round=1000,
                          evals=[(dval, "valid")], early_stopping_rounds=50, verbose_eval=False)
        probs = model.predict(dpred)
        actions = np.argmax(probs, axis=1)
        all_actions.extend(actions.tolist())
        all_probs.append(probs)
        i = chunk_end

    actions_arr = np.array(all_actions)
    probs_arr = np.vstack(all_probs)
    result = run_3action(actions_arr, oos_df)

    # ── 분석용 DataFrame 구성 ──
    analysis = oos_df[["date", "close"]].copy()
    analysis["action"] = actions_arr
    analysis["action_name"] = analysis["action"].map({0: "Buy", 1: "Hold", 2: "Sell"})
    analysis["prob_buy"] = probs_arr[:, 0]
    analysis["prob_hold"] = probs_arr[:, 1]
    analysis["prob_sell"] = probs_arr[:, 2]

    # 시장 상태 지표
    analysis["ret_1h"] = analysis["close"].pct_change()
    analysis["ret_24h"] = analysis["close"].pct_change(24)
    analysis["ret_7d"] = analysis["close"].pct_change(168)
    analysis["vol_24h"] = analysis["ret_1h"].rolling(24).std()
    analysis["vol_7d"] = analysis["ret_1h"].rolling(168).std()

    # 추세 판단 (7일 수익률 기준)
    analysis["trend"] = "sideways"
    analysis.loc[analysis["ret_7d"] > 0.03, "trend"] = "uptrend"
    analysis.loc[analysis["ret_7d"] < -0.03, "trend"] = "downtrend"

    # 변동성 구간
    vol_med = analysis["vol_24h"].median()
    analysis["vol_regime"] = "low_vol"
    analysis.loc[analysis["vol_24h"] > vol_med, "vol_regime"] = "high_vol"

    # 포지션 추적 (실제 보유 상태)
    position = []  # 0=현금, 1=보유
    pos = 0
    for a in actions_arr:
        if a == 0 and pos == 0: pos = 1
        elif a == 2 and pos == 1: pos = 0
        position.append(pos)
    analysis["position"] = position

    analysis = analysis.dropna()

    # ────────────────────────────────
    # 분석 1: 행동 분포 (전체)
    # ────────────────────────────────
    print(f"\n[1] 전체 행동 분포")
    for a in [0, 1, 2]:
        name = {0: "Buy", 1: "Hold", 2: "Sell"}[a]
        cnt = (actions_arr == a).sum()
        pct = cnt / len(actions_arr) * 100
        print(f"    {name}: {cnt} ({pct:.1f}%)")

    # ────────────────────────────────
    # 분석 2: 변동성 구간별 행동 분포
    # ────────────────────────────────
    print(f"\n[2] 변동성 구간별 행동 분포")
    for vr in ["low_vol", "high_vol"]:
        sub = analysis[analysis["vol_regime"] == vr]
        total = len(sub)
        if total == 0: continue
        buy_pct = (sub["action"] == 0).sum() / total * 100
        hold_pct = (sub["action"] == 1).sum() / total * 100
        sell_pct = (sub["action"] == 2).sum() / total * 100
        print(f"    {vr:>8s} (n={total:>4d}): Buy={buy_pct:5.1f}% Hold={hold_pct:5.1f}% Sell={sell_pct:5.1f}%")

    # ────────────────────────────────
    # 분석 3: 추세별 행동 분포
    # ────────────────────────────────
    print(f"\n[3] 추세별 행동 분포")
    for tr in ["uptrend", "sideways", "downtrend"]:
        sub = analysis[analysis["trend"] == tr]
        total = len(sub)
        if total == 0: continue
        buy_pct = (sub["action"] == 0).sum() / total * 100
        hold_pct = (sub["action"] == 1).sum() / total * 100
        sell_pct = (sub["action"] == 2).sum() / total * 100
        print(f"    {tr:>10s} (n={total:>4d}): Buy={buy_pct:5.1f}% Hold={hold_pct:5.1f}% Sell={sell_pct:5.1f}%")

    # ────────────────────────────────
    # 분석 4: 추세×변동성 교차 분석
    # ────────────────────────────────
    print(f"\n[4] 추세 × 변동성 교차 분석")
    for tr in ["uptrend", "sideways", "downtrend"]:
        for vr in ["low_vol", "high_vol"]:
            sub = analysis[(analysis["trend"] == tr) & (analysis["vol_regime"] == vr)]
            total = len(sub)
            if total < 10: continue
            buy_pct = (sub["action"] == 0).sum() / total * 100
            hold_pct = (sub["action"] == 1).sum() / total * 100
            sell_pct = (sub["action"] == 2).sum() / total * 100
            print(f"    {tr:>10s}+{vr:<8s} (n={total:>4d}): "
                  f"Buy={buy_pct:5.1f}% Hold={hold_pct:5.1f}% Sell={sell_pct:5.1f}%")

    # ────────────────────────────────
    # 분석 5: 포지션 보유 시간 분석
    # ────────────────────────────────
    print(f"\n[5] 포지션 보유 시간 분석")
    holding_periods = []
    in_position = False
    entry_idx = 0
    for idx, row in analysis.iterrows():
        if row["action"] == 0 and not in_position:
            in_position = True
            entry_idx = idx
        elif row["action"] == 2 and in_position:
            in_position = False
            holding_periods.append(idx - entry_idx)

    if holding_periods:
        hp = np.array(holding_periods)
        print(f"    총 매매 사이클: {len(hp)}회")
        print(f"    평균 보유 시간: {hp.mean():.1f}h")
        print(f"    중앙값 보유 시간: {np.median(hp):.1f}h")
        print(f"    최소/최대: {hp.min()}h / {hp.max()}h")
        print(f"    24h 이하: {(hp <= 24).sum()}회 ({(hp <= 24).sum()/len(hp)*100:.1f}%)")
        print(f"    24h~168h: {((hp > 24) & (hp <= 168)).sum()}회")
        print(f"    168h 이상: {(hp > 168).sum()}회")

    # ────────────────────────────────
    # 분석 6: 포지션 보유 비율 vs 시장 상태
    # ────────────────────────────────
    print(f"\n[6] 포지션 보유 비율 (시간 기준)")
    total_hours = len(analysis)
    in_position_hours = analysis["position"].sum()
    print(f"    전체 {total_hours}h 중 {in_position_hours}h 보유 ({in_position_hours/total_hours*100:.1f}%)")

    for tr in ["uptrend", "sideways", "downtrend"]:
        sub = analysis[analysis["trend"] == tr]
        if len(sub) == 0: continue
        pos_pct = sub["position"].mean() * 100
        print(f"    {tr:>10s}: {pos_pct:.1f}% 보유")

    for vr in ["low_vol", "high_vol"]:
        sub = analysis[analysis["vol_regime"] == vr]
        if len(sub) == 0: continue
        pos_pct = sub["position"].mean() * 100
        print(f"    {vr:>8s}: {pos_pct:.1f}% 보유")

    # ────────────────────────────────
    # 분석 7: 월별 행동 분포 + 매매 빈도
    # ────────────────────────────────
    print(f"\n[7] 월별 행동 분포")
    analysis["month"] = analysis["date"].dt.to_period("M")
    monthly = analysis.groupby("month").agg(
        buy_cnt=("action", lambda x: (x == 0).sum()),
        hold_cnt=("action", lambda x: (x == 1).sum()),
        sell_cnt=("action", lambda x: (x == 2).sum()),
        total=("action", "count"),
        avg_price=("close", "mean"),
        ret=("close", lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 1 else 0),
        avg_vol=("vol_24h", "mean"),
        pos_ratio=("position", "mean"),
    ).reset_index()

    print(f"    {'Month':>8s} | {'Price':>8s} | {'Ret':>6s} | {'Vol':>6s} | "
          f"{'Buy':>4s} {'Hold':>4s} {'Sell':>4s} | {'Pos%':>5s}")
    print(f"    {'-'*65}")
    for _, row in monthly.iterrows():
        print(f"    {str(row['month']):>8s} | ${row['avg_price']:>7.0f} | {row['ret']:>+5.1f}% | "
              f"{row['avg_vol']:.4f} | {row['buy_cnt']:>4.0f} {row['hold_cnt']:>4.0f} {row['sell_cnt']:>4.0f} | "
              f"{row['pos_ratio']*100:>4.1f}%")

    # ── 시각화: 월별 행동 분포 + 가격 ──
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), gridspec_kw={"height_ratios": [2, 1, 1]})

    # 상단: 가격 + 포지션 음영
    ax = axes[0]
    dates = analysis["date"].values
    prices = analysis["close"].values
    ax.plot(dates, prices, color="#333", linewidth=1)

    # 보유 구간 음영
    pos = analysis["position"].values
    in_pos = False
    start_idx = 0
    for j in range(len(pos)):
        if pos[j] == 1 and not in_pos:
            in_pos = True; start_idx = j
        elif pos[j] == 0 and in_pos:
            in_pos = False
            ax.axvspan(dates[start_idx], dates[j], alpha=0.15, color="#2196F3")
    if in_pos:
        ax.axvspan(dates[start_idx], dates[-1], alpha=0.15, color="#2196F3")

    ax.set_title(f"XGB 행동 분석 [{label}] | 파란 음영=보유 구간", fontsize=13, fontweight="bold")
    ax.set_ylabel("Price")
    sns.despine(ax=ax, left=True, bottom=True)

    # 중간: 행동 분포 (시간별 rolling)
    ax2 = axes[1]
    window = 168  # 7일
    buy_roll = (analysis["action"] == 0).rolling(window).mean().values
    sell_roll = (analysis["action"] == 2).rolling(window).mean().values
    ax2.fill_between(dates, 0, buy_roll, alpha=0.5, color="#2196F3", label="Buy ratio (7d)")
    ax2.fill_between(dates, 0, -sell_roll, alpha=0.5, color="#E53935", label="Sell ratio (7d)")
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_ylabel("Action ratio")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_ylim(-0.5, 0.5)
    sns.despine(ax=ax2, left=True, bottom=True)

    # 하단: 변동성
    ax3 = axes[2]
    ax3.plot(dates, analysis["vol_24h"].values, color="#9C27B0", linewidth=1, label="vol_24h")
    ax3.axhline(vol_med, color="gray", linestyle="--", alpha=0.5, label=f"median={vol_med:.4f}")
    ax3.set_ylabel("Volatility (24h)")
    ax3.legend(fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    sns.despine(ax=ax3, left=True, bottom=True)

    fig.tight_layout()
    path = os.path.join(exp_dir, f"behavior_{label.replace('~','_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [Chart] {path}")

print("\nDone!")
