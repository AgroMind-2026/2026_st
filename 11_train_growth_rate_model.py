# -*- coding: utf-8 -*-
"""
AgroMind - 생육 성장률 예측 모델 학습

목적:
    10_prepare_growth_stage_table.py로 만든
    train_table_growth_rate_full.csv를 사용해서
    "환경 조건 -> 생육 성장률" 모델을 학습한다.

    타겟은 하나가 아니라 여러 개(초장/엽수/엽장/엽폭/엽병장/관부직경의
    성장률)이므로, 지표별로 각각 모델을 학습한다.
    (추천 시스템에서는 이 여러 성장률 예측을 종합해서
     "이 환경이 전반적으로 성장에 도움이 되는가"를 판단하게 됨)

검증 방식:
    - 지금까지의 교훈(일반 KFold는 농가를 암기해 성능이 과대추정됨)을
      반영하여, 여기서도 GroupKFold(농가 단위)로 검증한다.
    - 다만 표본이 커봐야 1,700건 내외이고 농가당 데이터도 적어서,
      fold 수는 5와 (농가 수 // 3) 중 작은 값으로 자동 조정한다.
    - Pooled R2(out-of-fold 예측을 모아 한 번에 계산)를 주 지표로 삼는다.

피처 관련 주의사항:
    - '경과일수'(두 조사 사이 간격)는 성장률 계산식의 분모라서
      모델에 넣으면 그 자체로 노이즈를 줄여줄 수 있지만, 농가가
      "조절할 수 있는" 변수가 아니므로 추천 시스템에서 추론할 때는
      항상 표준값(예: 7일)으로 고정해서 넣어야 한다.
    - '생육조사_개체수'는 표본 크기 정보일 뿐 환경 요인이 아니므로
      피처에서 제외한다.
    - '화방착과여부', '수확중여부'는 True/False를 0/1로 변환해서 사용.

출력:
    - 지표별 GroupKFold 성능 요약 (터미널 출력 + CSV)
    - 지표별 변수 중요도 (RandomForest 기준)
    - 전체 데이터로 재학습한 최종 모델(지표별) -> joblib로 저장
      (추천 시스템 백엔드에서 이 모델 파일들을 그대로 불러와 사용 가능)
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# ============================================
# 설정
# ============================================

DATA_PATH = "data/train_table_growth_rate_full.csv"
MODEL_OUT_DIR = "growth_models"

TARGETS = [
    '초장_성장률',
    '엽수_성장률',
    '엽장_성장률',
    '엽폭_성장률',
    '엽병장_성장률',
    '관부직경_성장률',
]

os.makedirs(MODEL_OUT_DIR, exist_ok=True)

# ============================================
# 데이터 로드
# ============================================

df = pd.read_csv(DATA_PATH)

# 불리언성 컬럼 정리 (문자열 'True'/'False'로 읽힐 수 있어 안전하게 변환)
for c in ['화방착과여부', '수확중여부']:
    if df[c].dtype == object:
        df[c] = df[c].map({'True': True, 'False': False}).fillna(df[c])
    df[c] = df[c].astype(int)

# 피처 목록: 환경(rolling) + 재배정보 + 생육단계 신호
env_prefixes = [
    '온도_외부', '일사량_외부', '누적일사량_외부',
    '온도_내부', '상대습도_내부', '잔존 이산화탄소(CO2)'
]

env_feat_cols = [
    c for c in df.columns
    if any(c.startswith(p) for p in env_prefixes)
]

base_feat_cols = [
    '재식밀도',
    '식부면적',
    '정식후경과일',
    '화방착과여부',
    '수확중여부',
    '화방별착과수_개체평균',
    '경과일수',   # 추론 시에는 표준값(예: 7)으로 고정해서 넣을 것
]

feature_cols = env_feat_cols + base_feat_cols

print("사용 피처:", feature_cols)
print(f"\n전체 데이터: {df.shape}")

groups_all = df['농가ID']
n_farms = groups_all.nunique()
n_splits = max(2, min(5, n_farms // 3))

print(f"고유 농가 수: {n_farms}개 -> GroupKFold n_splits={n_splits}")

# ============================================
# 모델 정의
# ============================================

def make_models():
    return {
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        ),
    }


# ============================================
# 타겟별 GroupKFold 검증
# ============================================

summary_rows = []
importance_rows = []

for target in TARGETS:

    sub = df.dropna(subset=[target] + feature_cols).reset_index(drop=True)

    if len(sub) < 30:
        print(f"\n[{target}] 유효 데이터 {len(sub)}건 -> 너무 적어 스킵")
        continue

    X = sub[feature_cols]
    y = sub[target]
    groups = sub['농가ID']

    this_n_farms = groups.nunique()
    this_n_splits = max(2, min(n_splits, this_n_farms // 2, this_n_farms))

    gkf = GroupKFold(n_splits=this_n_splits)

    print(f"\n===== [{target}] (n={len(sub)}, 농가 {this_n_farms}개, fold {this_n_splits}) =====")

    pooled_store = {name: {'y_true': [], 'y_pred': []} for name in make_models()}

    for name, _ in make_models().items():
        for tr_idx, te_idx in gkf.split(X, y, groups=groups):
            models = make_models()
            model = models[name]

            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)

            pooled_store[name]['y_true'].extend(y_te.values.tolist())
            pooled_store[name]['y_pred'].extend(pred.tolist())

    for name in make_models():
        yt = np.array(pooled_store[name]['y_true'])
        yp = np.array(pooled_store[name]['y_pred'])

        pooled_r2 = r2_score(yt, yp)
        pooled_rmse = np.sqrt(mean_squared_error(yt, yp))
        pooled_mae = mean_absolute_error(yt, yp)

        print(
            f"  [{name}] Pooled R2={pooled_r2:.4f}  "
            f"RMSE={pooled_rmse:.4f}  MAE={pooled_mae:.4f}"
        )

        summary_rows.append({
            'target': target,
            'model': name,
            'n': len(sub),
            'n_farms': this_n_farms,
            'Pooled_R2': pooled_r2,
            'Pooled_RMSE': pooled_rmse,
            'Pooled_MAE': pooled_mae,
        })

    # ------------------------------------------------
    # 전체 데이터로 최종 모델 학습 (RandomForest 기준 채택)
    # -> 추천 시스템에서 그대로 불러 쓸 모델
    # ------------------------------------------------

    final_model = make_models()['RandomForest']
    final_model.fit(X, y)

    joblib.dump(
        final_model,
        os.path.join(MODEL_OUT_DIR, f"rf_{target}.joblib")
    )

    imp = pd.DataFrame({
        'target': target,
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    importance_rows.append(imp)

    print(f"  -> 최종 모델 저장: {MODEL_OUT_DIR}/rf_{target}.joblib")
    print(f"  변수 중요도 Top 5:")
    print(imp.head(5)[['feature', 'importance']].to_string(index=False))

# ============================================
# 결과 저장
# ============================================

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    'growth_model_cv_summary.csv',
    index=False,
    encoding='utf-8-sig'
)

if importance_rows:
    importance_df = pd.concat(importance_rows, ignore_index=True)
    importance_df.to_csv(
        'growth_model_feature_importance.csv',
        index=False,
        encoding='utf-8-sig'
    )

print("\n=== 전체 요약 (지표 x 모델별 Pooled R2) ===")
print(
    summary_df
    .pivot(index='target', columns='model', values='Pooled_R2')
    .to_string()
)

print(
    f"\n저장 완료:\n"
    f" - growth_model_cv_summary.csv (지표별 성능)\n"
    f" - growth_model_feature_importance.csv (지표별 변수 중요도)\n"
    f" - {MODEL_OUT_DIR}/rf_*.joblib (지표별 최종 모델, 추천 시스템에서 재사용)\n\n"
    "해석 가이드:\n"
    " - Pooled R2가 지표마다 다를 수 있음. 초장/엽수처럼 측정 오차가\n"
    "   상대적으로 큰 지표는 R2가 낮게 나올 수 있으니, RMSE/MAE도 함께 참고.\n"
    " - Ridge와 RandomForest/XGBoost 성능 차이가 크지 않다면, 표본이 작아\n"
    "   복잡한 모델의 이점이 크지 않다는 뜻이므로 Ridge(해석 쉬움)를\n"
    "   추천 시스템에 써도 무방함.\n"
    " - 변수 중요도에서 '경과일수'가 지나치게 높게 나오면, 그건 진짜 환경\n"
    "   신호가 아니라 측정 간격 노이즈를 반영하는 것일 수 있으니 주의."
)
