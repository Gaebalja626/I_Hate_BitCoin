"""
메인 실험 실행 스크립트
- 실행 전 설정 요약 출력 → 데이터 수집 → 피처 생성 → 학습 → 백테스트 → 시각화
- 모든 출력이 콘솔 + 로그 파일에 동시 기록
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

import config as cfg
from data_collector import load_or_fetch
from feature_engineer import create_features
from models import lgbm_model
from backtester import run as run_backtest, print_result
from visualizer import (
    plot_equity_curve,
    plot_drawdown,
    plot_feature_importance,
    plot_prediction_dist,
    plot_summary_dashboard,
    plot_threshold_sweep,
)
from experiment import create_experiment, save_metrics, list_experiments
from logger import setup_logger, add_experiment_log


def print_plan(log):
    plan = f"""
{'='*60}
  BTC/USDT LightGBM 트레이딩 실험
{'='*60}

[데이터]
  페어:        {cfg.PAIR}
  시간단위:    {cfg.TIMEFRAME} (1시간봉)
  학습 기간:   {cfg.TRAIN_START} ~ {cfg.TEST_START}
  테스트 기간: {cfg.TEST_START} ~ {cfg.DATA_END}

[피처 엔지니어링]
  보조지표:    {len(cfg.TECH_INDICATORS)}개
               {', '.join(cfg.TECH_INDICATORS[:8])}...
  예측 타겟:   {cfg.LOOKAHEAD}시간 후 상승/하락 (이진 분류)
  추가 피처:   수익률, 거래량변화, 캔들패턴, 가격위치 등

[모델]
  알고리즘:    LightGBM (GBDT)
  학습률:      {cfg.LGBM_PARAMS['learning_rate']}
  최대 라운드: {cfg.LGBM_NUM_BOOST} (early stop={cfg.LGBM_EARLY_STOP})
  검증 비율:   학습 데이터의 {cfg.LGBM_VAL_RATIO*100:.0f}% (시계열 후반부)
  정규화:      L1={cfg.LGBM_PARAMS['lambda_l1']}, L2={cfg.LGBM_PARAMS['lambda_l2']}

[백테스트]
  초기 자산:   ${cfg.INITIAL_BALANCE:,.0f}
  수수료:      {cfg.TRADING_FEE*100:.1f}%
  Threshold:   {cfg.THRESHOLDS}
  전략:        확률 > threshold → 매수(보유), 아니면 → 매도(현금)

[출력]
  실험 폴더:   results/<실험명>/
  차트:        수익곡선, 낙폭, 피처중요도, 예측분포, 종합대시보드
{'='*60}"""
    for line in plan.strip().split("\n"):
        log.info(line)


def main():
    log = setup_logger(name="lgbm_experiment")

    # ── 1) 계획 출력 ──
    print_plan(log)

    # ── 2) 데이터 수집 ──
    log.info("[STEP 1] 데이터 수집")
    raw = load_or_fetch(
        cfg.PAIR, cfg.TIMEFRAME, cfg.TRAIN_START, cfg.DATA_END, cfg.TECH_INDICATORS
    )

    # ── 3) 피처 생성 ──
    log.info("[STEP 2] 피처 엔지니어링")
    df, feature_cols = create_features(raw, cfg.TECH_INDICATORS, lookahead=cfg.LOOKAHEAD)

    # ── 4) Train/Test 분할 ──
    df["date"] = pd.to_datetime(df["date"])
    train_df = df[df["date"] < cfg.TEST_START].reset_index(drop=True)
    test_df = df[df["date"] >= cfg.TEST_START].reset_index(drop=True)
    log.info(f"  학습: {len(train_df)}행 ({cfg.TRAIN_START} ~ {cfg.TEST_START})")
    log.info(f"  테스트: {len(test_df)}행 ({cfg.TEST_START} ~ {cfg.DATA_END})")

    # ── 5) 학습 ──
    log.info("[STEP 3] 모델 학습")
    model, feat_imp, train_info = lgbm_model.train(train_df, feature_cols)
    log.info(f"  best_iter={train_info['best_iteration']} | val_loss={train_info['best_score']:.6f}")

    # ── 6) 실험 폴더 생성 ──
    exp_config = {
        "pair": cfg.PAIR,
        "timeframe": cfg.TIMEFRAME,
        "train_period": f"{cfg.TRAIN_START} ~ {cfg.TEST_START}",
        "test_period": f"{cfg.TEST_START} ~ {cfg.DATA_END}",
        "lookahead": cfg.LOOKAHEAD,
        "features": feature_cols,
        "lgbm_params": cfg.LGBM_PARAMS,
        "train_info": train_info,
    }
    exp_dir = create_experiment("lgbm", exp_config)
    add_experiment_log(log, exp_dir)
    log.info(f"[실험 폴더] {exp_dir}")

    # 모델 저장
    lgbm_model.save_model(model, os.path.join(exp_dir, "model.txt"))

    # 피처 중요도
    log.info("[피처 중요도 Top 10]")
    for _, row in feat_imp.head(10).iterrows():
        log.info(f"  {row['feature']:25s}  {row['importance']:.1f}")
    plot_feature_importance(feat_imp, os.path.join(exp_dir, "feature_importance.png"))

    # ── 7) 백테스트 ──
    log.info("[STEP 4] 백테스트 & 시각화")
    predictions = lgbm_model.predict(model, test_df, feature_cols)

    all_results = []
    for th in cfg.THRESHOLDS:
        result = run_backtest(predictions, test_df, threshold=th)
        all_results.append(result)

        log.info(f"--- threshold={th} ---")
        log.info(f"  수익률: {result['model_return_pct']:+.2f}% | B&H: {result['buyhold_return_pct']:+.2f}%")
        log.info(f"  MDD: {result['mdd_pct']:.2f}% | Sharpe: {result['sharpe_ratio']:.2f}")
        log.info(f"  정확도: {result['accuracy']} | 거래: {result['total_trades']}회")

        th_label = f"t{th}"
        save_metrics(exp_dir, result)
        plot_equity_curve(result, test_df, os.path.join(exp_dir, f"equity_{th_label}.png"))
        plot_drawdown(result, os.path.join(exp_dir, f"drawdown_{th_label}.png"))
        plot_prediction_dist(predictions, os.path.join(exp_dir, f"pred_dist_{th_label}.png"), threshold=th)
        plot_summary_dashboard(result, test_df, feat_imp, os.path.join(exp_dir, f"summary_{th_label}.png"))

    plot_threshold_sweep(all_results, os.path.join(exp_dir, "threshold_sweep.png"))

    # ── 8) 실험 목록 ──
    log.info("[STEP 5] 실험 이력")
    list_experiments()

    log.info(f"[완료] 결과 폴더: {exp_dir}")
    log.info(f"[로그] {exp_dir}/experiment.log")
    log.info(f"[전역 로그] results/all_experiments.log")


if __name__ == "__main__":
    main()
