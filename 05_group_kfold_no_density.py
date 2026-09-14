# -*- coding: utf-8 -*-
"""
AgroMind - 재식밀도 제외 + 농가 단위(Group) 5-Fold 교차검증

목적:
    04_group_kfold_cv.py 결과, GroupKFold(농가 단위 분할)에서 R2가
    음수로 크게 떨어짐 (-0.16 ~ -0.07). 재식밀도가 농가별 고정값이라
    모델이 "환경 -> 생산성" 관계 대신 "재식밀도 값 -> 농가 식별"을
    암기했을 가능성이 있음.

    이 스크립트는 재식밀도를 피처에서 제외한 뒤 동일하게 GroupKFold를
    돌려서, R2가 회복되는지(=재식밀도가 농가 proxy였는지) 확인한다.

    04_group_kfold_cv.py와 다른 점은 딱 하나:
        drop_cols 에 '재식밀도' 추가
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# ============================================
# 데이터 로드
# ============================================

df = pd.read_csv("data/train_table_combined.csv")

# 이상치 제거 (기존 스크립트와 동일한 기준)
df['출하량_단위면적'] = df['총출하량'] / df['식부면적']
df = df[df['출하량_단위면적'] > 0]

upper = df['출하량_단위면적'].quantile(0.99)
df = df[df['출하량_단위면적'] <= upper]

df = df.reset_index(drop=True)

target = '출하량_단위면적'

# ★ 변경점: '재식밀도'를 피처에서 제외
drop_cols = [
    '농가ID',
    '농가명',
    '출하일자',
    '기준일',
    '판매금액',
    '총출하량',
    '식부면적',
    '재식밀도',   # <- 농가별 고정값(농가 proxy 의심) 제외
    target
]

feature_cols = [
    c for c in df.columns
    if c not in drop_cols
]

print("사용된 피처 목록:")
print(feature_cols)

X = df[feature_cols].reset_index(drop=True)
y = df[target].reset_index(drop=True)
groups = df['농가ID'].reset_index(drop=True)

n_farms = groups.nunique()

print(
    f"\n전체 데이터: {len(df)}건 / "
    f"고유 농가 수: {n_farms}개"
)

if n_farms < 5:
    print(
        f"⚠ 경고: 농가 수({n_farms}개)가 fold 수(5)보다 적거나 같습니다. "
        f"n_splits를 농가 수 이하로 낮추세요."
    )

# ============================================
# 모델 정의 (기존과 동일)
# ============================================

model_builders = {

    'RandomForest': lambda: RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42
    ),

    'XGBoost': lambda: xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    ),

    'LightGBM': lambda: lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbosity=-1
    ),
}

# ============================================
# GroupKFold 교차검증 (재식밀도 제외 버전)
# ============================================

N_SPLITS = 5

gkf = GroupKFold(n_splits=N_SPLITS)

records = []
fold_farm_map = []

for name, builder in model_builders.items():

    fold_r2 = []
    fold_rmse = []
    fold_mae = []

    for fold_i, (tr_idx, te_idx) in enumerate(
        gkf.split(X, y, groups=groups),
        start=1
    ):

        X_tr = X.iloc[tr_idx]
        X_te = X.iloc[te_idx]

        y_tr = y.iloc[tr_idx]
        y_te = y.iloc[te_idx]

        train_farms = set(groups.iloc[tr_idx])
        test_farms = set(groups.iloc[te_idx])
        overlap = train_farms & test_farms

        assert len(overlap) == 0, (
            f"[{name} fold {fold_i}] 농가 누수 발생: {overlap}"
        )

        model = builder()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        r2 = r2_score(y_te, pred)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        mae = mean_absolute_error(y_te, pred)

        fold_r2.append(r2)
        fold_rmse.append(rmse)
        fold_mae.append(mae)

        records.append({
            'Model': name,
            'Fold': fold_i,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'n_test_farms': len(test_farms),
            'n_test_rows': len(te_idx),
        })

        if name == list(model_builders.keys())[0]:
            fold_farm_map.append({
                'Fold': fold_i,
                'test_farms': sorted(test_farms),
                'n_test_rows': len(te_idx),
            })

    print(f"\n[{name}] (GroupKFold, 농가 단위 분할, 재식밀도 제외)")
    print(f"  Fold별 R2 : {[round(v, 3) for v in fold_r2]}")
    print(
        f"  R2   평균±표준편차 : "
        f"{np.mean(fold_r2):.4f} ± {np.std(fold_r2):.4f}"
    )
    print(
        f"  RMSE 평균±표준편차 : "
        f"{np.mean(fold_rmse):.4f} ± {np.std(fold_rmse):.4f}"
    )
    print(
        f"  MAE  평균±표준편차 : "
        f"{np.mean(fold_mae):.4f} ± {np.std(fold_mae):.4f}"
    )

# ============================================
# 결과 저장
# ============================================

detail_df = pd.DataFrame(records)
detail_df.to_csv(
    'cv_results_detail_groupkfold_no_density.csv',
    index=False,
    encoding='utf-8-sig'
)

summary = (
    detail_df
    .groupby('Model')[['R2', 'RMSE', 'MAE']]
    .agg(['mean', 'std'])
)
summary.to_csv(
    'cv_results_summary_groupkfold_no_density.csv',
    encoding='utf-8-sig'
)

farm_map_df = pd.DataFrame(fold_farm_map)
farm_map_df.to_csv(
    'groupkfold_farm_assignment_no_density.csv',
    index=False,
    encoding='utf-8-sig'
)

print(
    "\n=== 요약 (GroupKFold, 재식밀도 제외, 5-Fold 평균 ± 표준편차) ==="
)
print(summary)

print(
    "\n해석 가이드:\n"
    " - R2가 04_group_kfold_cv.py(재식밀도 포함) 대비 뚜렷이 회복됐다면,\n"
    "   재식밀도가 순수 환경변수가 아니라 농가 식별자(proxy) 역할을\n"
    "   했다는 뜻입니다. -> 논문 한계점 서술을 뒷받침하는 실증 근거로 사용 가능.\n"
    " - 그래도 여전히 R2가 크게 음수라면, 재식밀도 외에\n"
    "   온도/CO2 등 환경변수의 농가별 설정범위(센서 캘리브레이션,\n"
    "   시설 특성)도 새 농가로 일반화를 막는 요인일 가능성이 있습니다."
)
