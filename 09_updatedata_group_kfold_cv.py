# -*- coding: utf-8 -*-
"""
AgroMind - 생육데이터 포함 + 농가 단위(Group) 5-Fold 교차검증

목적:
    01c_merge_growth_data.py로 만든 train_table_with_growth.csv
    (환경 + 재배정보 + 생육데이터)를 사용하여, 재식밀도는 그대로 두고
    생육 변수(초장, 엽수, 엽장, 엽폭, 엽병장, 관부직경, 화방별착과수 등)를
    추가로 넣었을 때 GroupKFold(농가 단위 분할) 성능이 개선되는지 확인한다.

    기존 04_group_kfold_cv.py와 비교 포인트:
        - 04번: 환경+재배정보만 (재식밀도 포함) -> Pooled/평균 R2 참고
        - 이 스크립트: 위 + 생육데이터 추가

전처리 관련 주의사항 (01c 실행 결과 기준):
    - 생육조사일자가 아예 없는 행(45건, 해당 시점 이전 조사기록 없음)은
      생육 컬럼이 NaN이 되므로 dropna()로 제거됨.
    - 생육조사경과일(기준일-생육조사일자)이 큰 행(2주 이상, 372건)은
      "오래된 생육값"이 붙어있다는 뜻이라 노이즈가 될 수 있음.
      -> STALE_THRESHOLD_DAYS로 필터링 옵션 제공 (기본은 사용 안 함,
         필요시 USE_STALE_FILTER=True로 켜서 비교 가능)
    - 화방별착과수_개체평균은 시간이 지날수록 누적되는 지표일 가능성이 있어
      타겟(출하량_단위면적)과 과도하게 직결된 '누수성 변수'는 아닌지 주의.
      우선은 포함하되, 필요시 제외하고 재실행해 비교 권장.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# ============================================
# 설정
# ============================================

DATA_PATH = "data/train_table_with_growth.csv"

# 생육조사경과일이 너무 큰(오래된) 행을 제외하고 싶으면 True로 변경
USE_STALE_FILTER = True
STALE_THRESHOLD_DAYS = 14

N_SPLITS = 5

# ============================================
# 데이터 로드
# ============================================

df = pd.read_csv(DATA_PATH)

df['출하일자'] = pd.to_datetime(df['출하일자'])
df['기준일'] = pd.to_datetime(df['기준일'])
if '생육조사일자' in df.columns:
    df['생육조사일자'] = pd.to_datetime(df['생육조사일자'])

# 생육값이 아예 안 붙은 행 제거 (이후 dropna에서도 걸러지지만 명시적으로 표시)
before_n = len(df)
df = df.dropna(subset=['초장', '엽수', '화방별착과수_개체평균'])
print(f"생육값 결측 제거: {before_n} -> {len(df)}건")

# (옵션) 너무 오래된 생육값이 붙은 행 제외
if USE_STALE_FILTER:
    before_n = len(df)
    df = df[df['생육조사경과일'] <= STALE_THRESHOLD_DAYS]
    print(
        f"생육조사경과일 > {STALE_THRESHOLD_DAYS}일 제외: "
        f"{before_n} -> {len(df)}건"
    )

# 이상치 제거 (기존과 동일한 기준)
df['출하량_단위면적'] = df['총출하량'] / df['식부면적']
df = df[df['출하량_단위면적'] > 0]

upper = df['출하량_단위면적'].quantile(0.99)
df = df[df['출하량_단위면적'] <= upper]

df = df.reset_index(drop=True)

target = '출하량_단위면적'

# 제외 컬럼: 식별자/날짜/타겟 계산 원재료 + 생육 원본 날짜(사용 불가)
drop_cols = [
    '농가ID',
    '농가명',
    '출하일자',
    '기준일',
    '생육조사일자',
    '판매금액',
    '총출하량',
    '식부면적',
    target,
]

feature_cols = [
    c for c in df.columns
    if c not in drop_cols
]

print("\n사용된 피처 목록 (재식밀도 포함 + 생육데이터 포함):")
print(feature_cols)

X = df[feature_cols].reset_index(drop=True)
y = df[target].reset_index(drop=True)
groups = df['농가ID'].reset_index(drop=True)

n_farms = groups.nunique()

print(
    f"\n전체 데이터: {len(df)}건 / "
    f"고유 농가 수: {n_farms}개"
)

if n_farms < N_SPLITS:
    print(
        f"⚠ 경고: 농가 수({n_farms}개)가 fold 수({N_SPLITS})보다 적거나 같습니다. "
        f"N_SPLITS를 낮추세요."
    )

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

# ============================================
# GroupKFold 교차검증
# ============================================

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

    print(f"\n[{name}] (GroupKFold, 농가 단위 분할, 생육데이터 포함)")
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
# Pooled 지표 (out-of-fold 예측을 모아 한 번에 계산 -> 더 안정적인 지표)
# ============================================

pooled_rows = []

# pooled 계산을 위해 fold별 예측을 다시 수집
# (records에는 fold별 요약 통계만 있으므로 별도로 out-of-fold 예측을 모음)
pooled_store = {name: {'y_true': [], 'y_pred': []} for name in model_builders}

for name, builder in model_builders.items():
    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        model = builder()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        pooled_store[name]['y_true'].extend(y_te.values.tolist())
        pooled_store[name]['y_pred'].extend(pred.tolist())

for name in model_builders:
    yt = np.array(pooled_store[name]['y_true'])
    yp = np.array(pooled_store[name]['y_pred'])

    pooled_rows.append({
        'Model': name,
        'Pooled_R2': r2_score(yt, yp),
        'Pooled_RMSE': np.sqrt(mean_squared_error(yt, yp)),
        'Pooled_MAE': mean_absolute_error(yt, yp),
        'n_total': len(yt),
    })

pooled_df = pd.DataFrame(pooled_rows)

print("\n=== Pooled 지표 (GroupKFold, 생육데이터 포함) ===")
print(pooled_df.to_string(index=False))

# ============================================
# 변수 중요도 (RandomForest 기준, 전체 데이터로 재학습해서 참고용으로 확인)
#   -> 생육 변수(초장/엽수 등)가 재식밀도/환경변수 대비 얼마나 중요한지 확인
# ============================================

rf_full = model_builders['RandomForest']()
rf_full.fit(X, y)

imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_full.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== 변수 중요도 (RandomForest, 전체 데이터 기준, Top 15) ===")
print(imp.head(15).to_string(index=False))

# ============================================
# 결과 저장
# ============================================

detail_df = pd.DataFrame(records)
detail_df.to_csv(
    'cv_results_detail_groupkfold_with_growth.csv',
    index=False,
    encoding='utf-8-sig'
)

summary = (
    detail_df
    .groupby('Model')[['R2', 'RMSE', 'MAE']]
    .agg(['mean', 'std'])
)
summary.to_csv(
    'cv_results_summary_groupkfold_with_growth.csv',
    encoding='utf-8-sig'
)

pooled_df.to_csv(
    'cv_results_pooled_groupkfold_with_growth.csv',
    index=False,
    encoding='utf-8-sig'
)

imp.to_csv(
    'feature_importance_with_growth.csv',
    index=False,
    encoding='utf-8-sig'
)

print("\n=== 요약 (GroupKFold, 농가 단위 분할, 생육데이터 포함, 5-Fold 평균 ± 표준편차) ===")
print(summary)

print(
    "\n비교 가이드:\n"
    " - 이 결과의 Pooled_R2를 생육데이터 없는 버전(04_group_kfold_cv.py를\n"
    "   pooled 방식으로 재계산한 값 또는 07/06b 스크립트의 GroupKFold 결과)과 비교하세요.\n"
    " - 변수 중요도 Top 15에 초장/엽수/화방별착과수 등이 상위에 오른다면,\n"
    "   생육 데이터가 실제로 유용한 정보를 제공한다는 뜻입니다.\n"
    " - 반대로 재식밀도/환경변수 중요도가 거의 그대로이고 생육변수가 하위권이라면,\n"
    "   생육데이터가 새 농가 일반화에는 큰 도움이 안 될 수 있습니다."
)