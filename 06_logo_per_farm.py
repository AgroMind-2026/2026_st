# -*- coding: utf-8 -*-
"""
AgroMind - LeaveOneGroupOut(농가 1개씩 제외) 검증
목적:
    GroupKFold(5-fold)의 R2가 크게 음수로 나온 게
    (a) 전반적으로 농가 간 일반화가 안 되는 구조적 문제인지,
    (b) 특정 농가 몇 곳의 이상치성 패턴 때문에 평균이 왜곡된 것인지
    구분하기 위해, 농가를 한 곳씩 test로 빼면서(=LeaveOneGroupOut)
    농가별 성능(R2, RMSE, MAE, 데이터 건수)을 개별적으로 확인한다.

    피처는 04_group_kfold_cv.py와 동일하게 재식밀도를 "포함"한
    원본 피처셋을 사용한다 (재식밀도 자체의 영향이 아니라,
    농가 단위 일반화가 어디서 깨지는지를 먼저 보기 위함).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# ============================================
# 데이터 로드
# ============================================

df = pd.read_csv("data/train_table_combined.csv")

df['출하량_단위면적'] = df['총출하량'] / df['식부면적']
df = df[df['출하량_단위면적'] > 0]

upper = df['출하량_단위면적'].quantile(0.99)
df = df[df['출하량_단위면적'] <= upper]

df = df.reset_index(drop=True)

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
print(f"전체 데이터: {len(df)}건 / 고유 농가 수: {n_farms}개")
print(f"-> LeaveOneGroupOut이므로 총 {n_farms}번의 fold가 실행됩니다.\n")

# ============================================
# 모델 정의
#   * LOGO는 fold 수(=농가 수)가 많아 3개 모델 다 돌리면 다소 오래 걸릴 수 있음.
#   * 필요시 model_builders에서 일부만 남겨도 됨.
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

logo = LeaveOneGroupOut()

records = []

for name, builder in model_builders.items():

    print(f"=== {name} 진행 중 ({n_farms} folds) ===")

    for fold_i, (tr_idx, te_idx) in enumerate(
        logo.split(X, y, groups=groups),
        start=1
    ):

        test_farm = groups.iloc[te_idx].iloc[0]

        X_tr = X.iloc[tr_idx]
        X_te = X.iloc[te_idx]
        y_tr = y.iloc[tr_idx]
        y_te = y.iloc[te_idx]

        model = builder()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        n_test = len(te_idx)

        # R2는 test 샘플 수가 1개거나 y_te 분산이 0에 가까우면
        # 정의상 불안정/undefined 하므로 별도 플래그를 남긴다.
        if n_test >= 2 and y_te.std() > 1e-8:
            r2 = r2_score(y_te, pred)
        else:
            r2 = np.nan

        rmse = np.sqrt(mean_squared_error(y_te, pred))
        mae = mean_absolute_error(y_te, pred)

        # 부호까지 포함한 평균 오차(bias): 양수면 과소예측, 음수면 과대예측 경향
        bias = float(np.mean(pred - y_te.values))

        records.append({
            'Model': name,
            '농가ID': test_farm,
            'n_test_rows': n_test,
            'y_mean': float(y_te.mean()),
            'y_std': float(y_te.std()) if n_test >= 2 else np.nan,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'Bias(예측-실제, 평균)': bias,
        })

detail_df = pd.DataFrame(records)

detail_df.to_csv(
    'logo_results_per_farm.csv',
    index=False,
    encoding='utf-8-sig'
)

# ============================================
# 모델별 전체 요약 (참고용)
# ============================================

overall_summary = (
    detail_df
    .groupby('Model')[['R2', 'RMSE', 'MAE']]
    .agg(['mean', 'std'])
)

print("\n=== 전체 요약 (농가별 R2/RMSE/MAE의 평균 ± 표준편차) ===")
print(overall_summary)

# ============================================
# 농가별로 RMSE가 가장 나쁜/좋은 곳을 모델별로 뽑아서 출력
#   -> "특정 농가 몇 곳 때문에 평균이 왜곡된 것"인지 눈으로 확인하기 위함
# ============================================

for name in model_builders.keys():

    sub = (
        detail_df[detail_df['Model'] == name]
        .sort_values('RMSE', ascending=False)
    )

    print(f"\n--- [{name}] RMSE 기준 가장 안 맞는 농가 Top 5 ---")
    print(
        sub[
            ['농가ID', 'n_test_rows', 'y_mean', 'RMSE', 'MAE', 'R2', 'Bias(예측-실제, 평균)']
        ]
        .head(5)
        .to_string(index=False)
    )

    print(f"\n--- [{name}] RMSE 기준 가장 잘 맞는 농가 Top 5 ---")
    print(
        sub[
            ['농가ID', 'n_test_rows', 'y_mean', 'RMSE', 'MAE', 'R2', 'Bias(예측-실제, 평균)']
        ]
        .tail(5)
        .to_string(index=False)
    )

print(
    "\n해석 가이드:\n"
    " - 소수 농가의 RMSE가 나머지 대비 압도적으로 크다면(예: 다른 농가의 3~5배 이상),\n"
    "   'GroupKFold R2가 음수인 건 전반적 문제가 아니라 특정 농가 이상치 때문'일 가능성이 높습니다.\n"
    "   -> 해당 농가의 재식밀도/재배정보/이상치 여부를 개별 점검해보세요.\n"
    " - 반대로 대부분 농가에서 고르게 RMSE가 높고 R2가 음수/0 근처라면,\n"
    "   특정 농가 문제가 아니라 '새로운 농가로의 일반화 자체가 구조적으로 어렵다'는 뜻이며,\n"
    "   피처 엔지니어링(농가 내 정규화 등)이나 더 많은 농가/데이터 확보가 필요합니다.\n"
    " - Bias 열이 특정 방향(계속 양수 또는 계속 음수)으로 치우쳐 있다면,\n"
    "   모델이 test 농가를 '항상 과소/과대예측'하는 경향이 있다는 뜻이므로,\n"
    "   그 농가의 재식밀도/생산성 수준이 학습 데이터 분포에서 벗어나 있는지도 확인해보세요."
)
