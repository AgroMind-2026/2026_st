# -*- coding: utf-8 -*-
"""
AgroMind - 모델링 및 변수중요도 분석
목표: 환경 변수(과거 7일 평균)로 딸기 일별 출하량(총출하량) 예측
      -> XGBoost / LightGBM / RandomForest 성능 비교 + 변수 중요도 분석
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# 한글 폰트 설정 (나눔고딕 없을 시 기본 폰트로 대체됨)
plt.rcParams['axes.unicode_minus'] = False
for f in fm.fontManager.ttflist:
    if 'Nanum' in f.name or 'Malgun' in f.name:
        plt.rcParams['font.family'] = f.name
        break

# 학습 데이터 불러오기
df = pd.read_csv('train_table_combined.csv')
print(f"학습 데이터: {df.shape}")

# -----------------------------
# 0. 농가 규모 효과 제거: 단위면적당 출하량(kg/m^2)으로 타겟 재정의
#    (식부면적이 크면 총출하량도 커지는 confound를 제거하여
#     순수하게 "환경조건 -> 생산성" 관계를 보기 위함)
# -----------------------------
df['출하량_단위면적'] = df['총출하량'] / df['식부면적']

# -----------------------------
# 1. 이상치 제거 (0 이하이거나 상위 1% 극단값)
# -----------------------------
df = df[df['출하량_단위면적'] > 0]
upper = df['출하량_단위면적'].quantile(0.99)
df = df[df['출하량_단위면적'] <= upper]
print(f"이상치 제거 후: {df.shape}")

# -----------------------------
# 2. 피처/타겟 정의
#    - 식부면적, 총출하량, 판매금액은 타겟 계산에 이미 쓰였거나
#      규모 confound이므로 피처에서 제외
#    - 재식밀도는 재배 "밀도"(관리 방식) 정보이므로 피처로 유지
# -----------------------------
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

X = df[feature_cols]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# -----------------------------
# 3. 모델 학습 & 평가
# -----------------------------
models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42
    ),

    'XGBoost': xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    ),

    'LightGBM': lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbosity=-1
    ),
}

results = []
preds_dict = {}
importances_dict = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    preds_dict[name] = pred

    rmse = np.sqrt(
        mean_squared_error(y_test, pred)
    )

    mae = mean_absolute_error(
        y_test,
        pred
    )

    r2 = r2_score(
        y_test,
        pred
    )

    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    })

    if hasattr(model, 'feature_importances_'):
        importances_dict[name] = model.feature_importances_

results_df = pd.DataFrame(results)

print("\n=== 모델 성능 비교 ===")
print(results_df.to_string(index=False))

# 결과 저장
results_df.to_csv(
    'model_results.csv',
    index=False,
    encoding='utf-8-sig'
)

# -----------------------------
# 4. 변수 중요도 (가장 성능 좋은 모델 기준)
# -----------------------------
best_model_name = (
    results_df
    .sort_values('R2', ascending=False)
    .iloc[0]['Model']
)

print(f"\n최고 성능 모델: {best_model_name}")

imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances_dict[best_model_name]
}).sort_values(
    'importance',
    ascending=False
)

print("\n=== 변수 중요도 (Top 10) ===")
print(
    imp.head(10).to_string(index=False)
)

imp.to_csv(
    'feature_importance.csv',
    index=False,
    encoding='utf-8-sig'
)

# -----------------------------
# 5. 시각화 1: 변수 중요도 Bar Chart
# -----------------------------
plt.figure(figsize=(8, 6))

top_n = imp.head(12)

sns.barplot(
    data=top_n,
    y='feature',
    x='importance',
    color='#2E8B57'
)

plt.title(
    f'{best_model_name} 변수 중요도 (Top 12)',
    fontsize=13
)

plt.xlabel('Importance')
plt.ylabel('')

plt.tight_layout()

plt.savefig(
    'fig_feature_importance.png',
    dpi=200
)

plt.close()

# -----------------------------
# 6. 시각화 2: 상관관계 히트맵
#    (주요 환경변수 mean 값 + 타겟)
# -----------------------------
mean_cols = [
    c for c in feature_cols
    if c.endswith('_mean')
] + [
    '정식후경과일',
    '재식밀도'
]

corr_df = df[
    mean_cols + [target]
].corr()

plt.figure(figsize=(9, 7))

sns.heatmap(
    corr_df,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    square=True,
    cbar_kws={'shrink': 0.8}
)

plt.title(
    '환경 변수 - 출하량 상관관계',
    fontsize=13
)

plt.tight_layout()

plt.savefig(
    'fig_correlation_heatmap.png',
    dpi=200
)

plt.close()

# -----------------------------
# 7. 시각화 3: 실제값 vs 예측값 (최고 모델)
# -----------------------------
plt.figure(figsize=(6, 6))

plt.scatter(
    y_test,
    preds_dict[best_model_name],
    alpha=0.4,
    color='#2E8B57',
    s=20
)

lims = [
    min(
        y_test.min(),
        preds_dict[best_model_name].min()
    ),
    max(
        y_test.max(),
        preds_dict[best_model_name].max()
    )
]

plt.plot(
    lims,
    lims,
    'r--',
    linewidth=1
)

plt.xlabel(
    '실제 단위면적당 출하량 (kg/m²)'
)

plt.ylabel(
    '예측 단위면적당 출하량 (kg/m²)'
)

plt.title(
    f'{best_model_name}: 실제 vs 예측',
    fontsize=13
)

plt.tight_layout()

plt.savefig(
    'fig_actual_vs_pred.png',
    dpi=200
)

plt.close()

print(
    "\n모든 결과물 저장 완료: "
    "model_results.csv, "
    "feature_importance.csv, "
    "fig_*.png"
)