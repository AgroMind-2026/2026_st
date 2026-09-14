# -*- coding: utf-8 -*-
"""
AgroMind - 5-Fold 교차검증으로 모델 성능 안정성 확인
(XGBoost vs LightGBM vs RandomForest, 단일 train/test split의 우연성 배제)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# 학습 데이터 불러오기
df = pd.read_csv("data/train_table_combined.csv")

# 이상치 제거 (기존과 동일한 기준)
df['출하량_단위면적'] = df['총출하량'] / df['식부면적']

df = df[df['출하량_단위면적'] > 0]

upper = df['출하량_단위면적'].quantile(0.99)

df = df[
    df['출하량_단위면적'] <= upper
]

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

print(
    f"전체 데이터: {len(df)}건 "
    f"(5-fold이므로 매 fold당 학습 {len(df)*4//5}건 / "
    f"검증 {len(df)//5}건 내외)"
)

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

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

records = []

for name, builder in model_builders.items():

    fold_r2 = []
    fold_rmse = []
    fold_mae = []

    for fold_i, (tr_idx, te_idx) in enumerate(
        kf.split(X),
        start=1
    ):

        X_tr = X.iloc[tr_idx]
        X_te = X.iloc[te_idx]

        y_tr = y.iloc[tr_idx]
        y_te = y.iloc[te_idx]

        model = builder()

        model.fit(
            X_tr,
            y_tr
        )

        pred = model.predict(
            X_te
        )

        r2 = r2_score(
            y_te,
            pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_te,
                pred
            )
        )

        mae = mean_absolute_error(
            y_te,
            pred
        )

        fold_r2.append(r2)
        fold_rmse.append(rmse)
        fold_mae.append(mae)

        records.append({
            'Model': name,
            'Fold': fold_i,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae
        })

    print(f"\n[{name}]")

    print(
        f"  Fold별 R2 : "
        f"{[round(v, 3) for v in fold_r2]}"
    )

    print(
        f"  R2   평균±표준편차 : "
        f"{np.mean(fold_r2):.4f} ± "
        f"{np.std(fold_r2):.4f}"
    )

    print(
        f"  RMSE 평균±표준편차 : "
        f"{np.mean(fold_rmse):.4f} ± "
        f"{np.std(fold_rmse):.4f}"
    )

    print(
        f"  MAE  평균±표준편차 : "
        f"{np.mean(fold_mae):.4f} ± "
        f"{np.std(fold_mae):.4f}"
    )


# -----------------------------
# Fold별 상세 결과 저장
# -----------------------------

detail_df = pd.DataFrame(records)

detail_df.to_csv(
    'cv_results_detail.csv',
    index=False,
    encoding='utf-8-sig'
)


# -----------------------------
# 모델별 평균 / 표준편차 저장
# -----------------------------

summary = (
    detail_df
    .groupby('Model')[
        ['R2', 'RMSE', 'MAE']
    ]
    .agg(['mean', 'std'])
)

summary.to_csv(
    'cv_results_summary.csv',
    encoding='utf-8-sig'
)

print(
    "\n=== 요약 "
    "(5-Fold 평균 ± 표준편차) ==="
)

print(summary)