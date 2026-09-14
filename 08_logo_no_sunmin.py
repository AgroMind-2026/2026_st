# -*- coding: utf-8 -*-
"""
AgroMind - 일사량_외부_min 제외 + LeaveOneGroupOut(농가 1개씩 제외) 검증

배경:
    농가별로 '일사량_외부_min' 값을 뜯어본 결과, 27개 농가 중 24개는
    모든 행이 정확히 0인 반면 2023_19/2023_32/2023_33 등 소수 농가는
    거의 항상 0이 아닌 값을 가짐 (중간값 없이 이분법적).
    실제 일조량이라면 야간엔 0이어야 정상이므로, 이는 환경 신호가
    아니라 센서 캘리브레이션/단위 차이로 인한 "농가 식별 변수"에
    가까운 것으로 판단됨.

    -> 이 피처를 제외하면 (특히 소수파 그룹에 속한 농가가 LOGO test로
       빠질 때) 예측 왜곡이 줄어드는지 확인한다.

기준 피처셋: 06b_logo_with_density_pooled.py와 동일 (재식밀도 포함)
변경점: drop_cols 에 '일사량_외부_min' 추가
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

# ★ 변경점: '일사량_외부_min' 제외 (재식밀도는 그대로 유지)
drop_cols = [
    '농가ID',
    '농가명',
    '출하일자',
    '기준일',
    '판매금액',
    '총출하량',
    '식부면적',
    '일사량_외부_min',   # <- 농가 식별 변수로 의심되어 제외 
]

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
    'logo_results_per_farm_no_sunmin.csv',
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
    'logo_pooled_summary_no_sunmin.csv',
    index=False,
    encoding='utf-8-sig'
)

print("\n=== Pooled 지표 (일사량_외부_min 제외, 재식밀도 포함) ===")
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
#   -> 특히 2023_19, 2023_32, 2023_33 (일사량_min 소수파 그룹)의
#      순위가 이전보다 개선됐는지 확인 포인트
# ============================================

watch_farms = {'2023_19', '2023_32', '2023_33', '2022_2', '2023_31'}

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

    print(f"\n--- [{name}] 일사량_min 소수파 그룹 농가 성적 (참고) ---")
    watch_sub = sub[sub['농가ID'].isin(watch_farms)]
    print(watch_sub[cols].to_string(index=False))

print(
    "\n해석 가이드:\n"
    " - 이 스크립트의 Pooled_R2를 06b_logo_with_density_pooled.py\n"
    "   (일사량_외부_min 포함) 결과와 비교하세요.\n"
    "     · RandomForest: -0.183 -> ?\n"
    "     · XGBoost     : +0.087 -> ?\n"
    "     · LightGBM    : -0.112 -> ?\n"
    "   특히 XGBoost의 양수 R2가 '일사량_외부_min을 이용한 농가 식별' 덕분이었다면\n"
    "   이 값이 크게 떨어질 것이고, 진짜 환경 신호(재식밀도 등) 덕분이었다면\n"
    "   크게 변하지 않거나 오히려 개선될 것입니다.\n"
    " - '일사량_min 소수파 그룹 농가 성적'에서 2023_19/2023_32/2023_33의\n"
    "   RMSE/상대오차가 이전 실험보다 줄었다면, 해당 피처 제거가\n"
    "   의도한 효과(농가 식별 신호 제거)를 낸 것으로 볼 수 있습니다."
)
