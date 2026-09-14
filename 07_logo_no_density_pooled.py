# -*- coding: utf-8 -*-
"""
AgroMind - 재식밀도 제외 + LeaveOneGroupOut(농가 1개씩 제외) 검증

목적:
    06_logo_per_farm.py(재식밀도 포함)에서 농가별 R2 평균이 -6 근처로
    보였는데, 이는 일부 저분산 농가(예: 2022_42, 2023_3)에서 R2가
    수학적으로 폭발한 통계적 인공물임을 확인함. 이번 스크립트는:
      1) 재식밀도를 피처에서 제외하고
      2) "농가별 R2 평균"이 아니라, 전체 out-of-fold 예측을 모아
         한 번에 계산하는 "pooled R2"를 함께 제공하여
    재식밀도 제외가 실제로 일반화 성능에 도움이 되는지를
    더 안정적인 지표로 판단할 수 있게 한다.

06_logo_per_farm.py와의 차이:
    - drop_cols 에 '재식밀도' 추가
    - 전체 pooled R2/RMSE/MAE 계산 추가 (농가별 평균의 왜곡 문제 보완)
    - 상대오차(RMSE / 농가 평균 출하량) 컬럼 추가 -> 저분산 농가 왜곡 방지
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

# ★ 변경점: '재식밀도'를 피처에서 제외
drop_cols = [
    '농가ID',
    '농가명',
    '출하일자',
    '기준일',
    '판매금액',
    '총출하량',
    '식부면적',
    '재식밀도',   # <- 제외
    target
]

feature_cols = [
    c for c in df.columns
    if c not in drop_cols
]

print("사용된 피처 목록 (재식밀도 제외):")
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
# pooled 계산을 위해 모델별로 모든 out-of-fold 예측/실제값을 모아둠
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

        # 개별 농가 R2는 여전히 참고용으로만 계산 (분산 0에 가까우면 NaN)
        if n_test >= 2 and y_te.std() > 1e-8:
            r2 = r2_score(y_te, pred)
        else:
            r2 = np.nan

        rmse = np.sqrt(mean_squared_error(y_te, pred))
        mae = mean_absolute_error(y_te, pred)
        bias = float(np.mean(pred - y_te.values))

        y_mean = float(y_te.mean())
        # 상대오차: 절대적 RMSE 대신, 그 농가 평균 출하량 대비 비율로 확인
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
    'logo_results_per_farm_no_density.csv',
    index=False,
    encoding='utf-8-sig'
)

# ============================================
# Pooled 지표: 모든 out-of-fold 예측을 한 번에 모아 계산
#   -> 저분산 농가 하나 때문에 R2가 폭발하는 문제를 피할 수 있음
#   -> "27개 농가를 하나씩 안 보고 학습했을 때 전체적으로 얼마나 맞았나"를
#      가장 정직하게 보여주는 지표
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
    'logo_pooled_summary_no_density.csv',
    index=False,
    encoding='utf-8-sig'
)

print("\n=== Pooled 지표 (전체 out-of-fold 예측을 한 번에 계산, 재식밀도 제외) ===")
print(pooled_df.to_string(index=False))

# 참고용: 농가별 R2 단순평균도 같이 출력 (인공물 가능성 있음을 명시)
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
# 상대오차(RMSE/평균) 기준으로 가장 안 맞는/잘 맞는 농가 확인
#   -> 절대 RMSE보다 저분산·저평균 농가의 왜곡을 줄여서 비교 가능
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
    " - Pooled_R2가 06_logo_per_farm.py(재식밀도 포함)의 Pooled_R2보다 높다면,\n"
    "   재식밀도 제외가 실제로 새 농가 일반화에 도움이 된다는 뜻입니다.\n"
    "   (같은 방식으로 06번 결과도 pooled로 재계산해서 비교하는 걸 추천)\n"
    " - '농가별 R2 단순평균'은 저분산 농가 때문에 극단값에 끌려가기 쉬우므로,\n"
    "   Pooled_R2를 주된 판단 기준으로 삼으세요.\n"
    " - 상대오차(RMSE/평균) 기준 Top5는 농가 규모와 무관하게 '비율로 얼마나 틀렸는지'를\n"
    "   보여주므로, 여기 반복적으로 등장하는 농가는 재배정보/이상치를 개별 점검해볼 가치가 있습니다."
)
