# -*- coding: utf-8 -*-
"""
AgroMind - 농가 단위(Group) 5-Fold 교차검증
목표: 동일 농가 데이터가 train/test에 동시에 섞이지 않도록
      GroupKFold(groups=농가ID)로 분할하여, 모델이 "농가 고유 패턴"을
      외워서 성능이 과대추정되는 것을 방지하고 보다 엄격한 일반화
      성능을 확인한다.

기존 03_cross_validation.py와 차이점:
    - KFold(shuffle=True) -> GroupKFold
    - 같은 농가의 row는 항상 같은 fold(전부 train 또는 전부 test)에만 배치됨
    - fold별로 어떤 농가가 test에 들어갔는지도 함께 출력/저장
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

target = '농가ID'  # placeholder, 아래에서 재정의됨 (실수 방지용 아님, 무시)
target = '출하량_단위면적'

drop_cols = [
    '농가ID',
    '농가명',
    '출하일자',
    '기준일',
    '판매금액',
    '총출하량',
    '식부면적',
    target
]

feature_cols = [
    c for c in df.columns
    if c not in drop_cols
]

X = df[feature_cols].reset_index(drop=True)
y = df[target].reset_index(drop=True)
groups = df['농가ID'].reset_index(drop=True)

n_farms = groups.nunique()

print(
    f"전체 데이터: {len(df)}건 / "
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
# GroupKFold 교차검증
#   - shuffle 옵션이 없음 (그룹 배정은 결정적)
#   - 같은 농가ID를 가진 행은 항상 같은 fold에만 속함
# ============================================

N_SPLITS = 5

gkf = GroupKFold(n_splits=N_SPLITS)

records = []
fold_farm_map = []  # 어떤 농가가 어떤 fold의 test에 들어갔는지 기록

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

        # 검증용: train/test 농가 집합이 겹치지 않는지 확인
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

        # 모델별로 fold-농가 매핑은 한 번만 기록하면 충분 (모델과 무관)
        if name == list(model_builders.keys())[0]:
            fold_farm_map.append({
                'Fold': fold_i,
                'test_farms': sorted(test_farms),
                'n_test_rows': len(te_idx),
            })

    print(f"\n[{name}] (GroupKFold, 농가 단위 분할)")
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
    'cv_results_detail_groupkfold.csv',
    index=False,
    encoding='utf-8-sig'
)

summary = (
    detail_df
    .groupby('Model')[['R2', 'RMSE', 'MAE']]
    .agg(['mean', 'std'])
)
summary.to_csv(
    'cv_results_summary_groupkfold.csv',
    encoding='utf-8-sig'
)

farm_map_df = pd.DataFrame(fold_farm_map)
farm_map_df.to_csv(
    'groupkfold_farm_assignment.csv',
    index=False,
    encoding='utf-8-sig'
)

print("\n=== 요약 (GroupKFold, 농가 단위 분할, 5-Fold 평균 ± 표준편차) ===")
print(summary)

print(
    "\n참고: 일반 KFold(03_cross_validation.py) 결과와 비교했을 때 "
    "R2가 눈에 띄게 낮아진다면, 기존 모델이 '농가 고유 특성'을 "
    "일부 외워서 성능이 과대추정됐을 가능성을 시사합니다."
)
