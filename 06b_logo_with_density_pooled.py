# -*- coding: utf-8 -*-
"""
AgroMind - LeaveOneGroupOut(농가 1개씩 제외) 검증 + Pooled 지표
(재식밀도 "포함" 버전 - 06_logo_per_farm.py에 pooled R2 로직만 추가)

목적:
    06_logo_per_farm.py에서는 "농가별 R2"를 단순 평균했는데,
    저분산 농가(예: 2022_42, 2023_3) 때문에 R2가 수학적으로 폭발해
    평균 R2가 -6 근처로 왜곡됨을 확인함.

    이 스크립트는 재식밀도를 "포함"한 원본 피처셋을 그대로 두고,
    07_logo_no_density_pooled.py와 동일하게 Pooled R2(전체 out-of-fold
    예측을 한 번에 모아 계산)를 추가로 산출한다.

    -> 이렇게 하면 07번(재식밀도 제외)의 Pooled_R2와 직접 비교해서,
       재식밀도 제외가 실제로 새 농가 일반화에 도움이 되는지
       공정하게 판단할 수 있다.

06_logo_per_farm.py와의 차이:
    - drop_cols는 원본 그대로 (재식밀도 포함)
    - 전체 pooled R2/RMSE/MAE 계산 추가
    - 상대오차(RMSE / 농가 평균 출하량) 컬럼 추가
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

# 재식밀도 "포함" (07번과의 유일한 실질적 차이)
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

print("사용된 피처 목록 (재식밀도 포함):")
print(feature_cols)

X = df[feature_cols].reset_index(drop=True)
y = df[target].reset_index(drop=True)
groups = df['농가ID'].reset_index(drop=True)

n_farms = groups.nunique()
print(f"\n전체 데이터: {len(df)}건 / 고유 농가 수: {n_farms}개")
print(f"-> LeaveOneGroupOut이므로 총 {n_farms}번의 fold가 실행됩니다.\n")

# ============================================
# 모델 정의
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
pooled_store = {name: {'y_true': [], 'y_pred': []} for name in model_builders}

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

        if n_test >= 2 and y_te.std() > 1e-8:
            r2 = r2_score(y_te, pred)
        else:
            r2 = np.nan

        rmse = np.sqrt(mean_squared_error(y_te, pred))
        mae = mean_absolute_error(y_te, pred)
        bias = float(np.mean(pred - y_te.values))

        y_mean = float(y_te.mean())
        rel_rmse = rmse / y_mean if y_mean > 1e-8 else np.nan

        records.append({
            'Model': name,
            '농가ID': test_farm,
            'n_test_rows': n_test,
            'y_mean': y_mean,
            'y_std': float(y_te.std()) if n_test >= 2 else np.nan,
            'R2': r2,
            'RMSE': rmse,
            'RMSE/평균(상대오차)': rel_rmse,
            'MAE': mae,
            'Bias(예측-실제, 평균)': bias,
        })

        pooled_store[name]['y_true'].extend(y_te.values.tolist())
        pooled_store[name]['y_pred'].extend(pred.tolist())

detail_df = pd.DataFrame(records)

detail_df.to_csv(
    'logo_results_per_farm_with_density.csv',
    index=False,
    encoding='utf-8-sig'
)

# ============================================
# Pooled 지표
# ============================================

pooled_rows = []

for name in model_builders:

    yt = np.array(pooled_store[name]['y_true'])
    yp = np.array(pooled_store[name]['y_pred'])

    pooled_r2 = r2_score(yt, yp)
    pooled_rmse = np.sqrt(mean_squared_error(yt, yp))
    pooled_mae = mean_absolute_error(yt, yp)

    pooled_rows.append({
        'Model': name,
        'Pooled_R2': pooled_r2,
        'Pooled_RMSE': pooled_rmse,
        'Pooled_MAE': pooled_mae,
        'n_total': len(yt),
    })

pooled_df = pd.DataFrame(pooled_rows)
pooled_df.to_csv(
    'logo_pooled_summary_with_density.csv',
    index=False,
    encoding='utf-8-sig'
)

print("\n=== Pooled 지표 (전체 out-of-fold 예측을 한 번에 계산, 재식밀도 포함) ===")
print(pooled_df.to_string(index=False))

naive_avg = (
    detail_df
    .groupby('Model')[['R2', 'RMSE', 'MAE']]
    .agg(['mean', 'std'])
)

print(
    "\n(참고, 왜곡 가능성 있음) 농가별 R2 단순 평균 ± 표준편차:"
)
print(naive_avg)

# ============================================
# 상대오차(RMSE/평균) 기준 Top5
# ============================================

for name in model_builders.keys():

    sub = (
        detail_df[detail_df['Model'] == name]
        .sort_values('RMSE/평균(상대오차)', ascending=False)
    )

    cols = [
        '농가ID', 'n_test_rows', 'y_mean',
        'RMSE', 'RMSE/평균(상대오차)', 'MAE', 'R2', 'Bias(예측-실제, 평균)'
    ]

    print(f"\n--- [{name}] 상대오차(RMSE/평균) 기준 가장 안 맞는 농가 Top 5 ---")
    print(sub[cols].head(5).to_string(index=False))

    print(f"\n--- [{name}] 상대오차(RMSE/평균) 기준 가장 잘 맞는 농가 Top 5 ---")
    print(sub[cols].tail(5).to_string(index=False))

print(
    "\n해석 가이드:\n"
    " - 이 스크립트의 Pooled_R2(재식밀도 포함)와 07_logo_no_density_pooled.py의\n"
    "   Pooled_R2(재식밀도 제외)를 나란히 비교하세요.\n"
    "     · 포함 버전이 더 높다면 -> 재식밀도가 새 농가 일반화에도 실제로 도움되는 신호.\n"
    "     · 제외 버전이 더 높다면 -> 재식밀도가 농가 식별자(proxy) 역할을 했을 가능성.\n"
    "     · 둘 다 0 근처거나 음수라면 -> 재식밀도 유무와 무관하게, 이 피처셋만으로는\n"
    "       처음 보는 농가에 대한 일반화 자체가 근본적으로 어렵다는 뜻."
)
