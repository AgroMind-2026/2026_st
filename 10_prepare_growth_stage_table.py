# -*- coding: utf-8 -*-
"""
AgroMind - 환경->생육 성장률 학습 테이블 생성 스크립트 (전체 기간)

목적:
    "환경 -> 생육 성장" 관계를 학습하기 위한 테이블을 만든다.
    기존 train_table_combined.csv는 "출하(판매)가 있는 날"만 존재해서
    정식~첫 출하 사이(생육 초반)의 환경-생육 관계를 학습할 수 없었다.
    이 스크립트는 생육 조사 데이터를 기준으로, 각 조사일 사이의
    "성장 속도(성장률)"를 계산하고, 그 기간의 환경(rolling 7일) 값을
    붙여서 학습 테이블을 만든다.

    처음에는 '수확 시작 전(pre-harvest)' 구간만 쓰려고 했으나,
    딸기는 수확이 시작된 이후에도 계속 생장(새 잎/화방 발생, 관부
    비대 등)하고, pre-harvest만 쓰면 표본이 190건까지 줄어들어
    학습이 불안정해지므로 -> 필터링 없이 "전체 기간"을 사용하고,
    대신 '수확중여부' 플래그를 피처로 남긴다. 즉 생육단계를 완전히
    분리된 모델로 나누지 않고, 하나의 '환경->성장률' 모델이
    정식후경과일/화방착과여부/수확중여부를 함께 보고 생육 단계별
    차이를 자동으로 학습하도록 한다. 추천 시스템에서는 이 모델의
    출력(성장률 예측)을 생육기/수확기 구분 없이 "지금 환경이 성장에
    도움되는 방향인가"를 판단하는 데 공통으로 활용할 수 있다.

사전 확인된 데이터 특성 (중요):
    - '작기' 컬럼은 이번 데이터셋에서 전 농가 값이 1로 고정되어 있어
      재배 사이클 구분에 사용할 수 없음 (그래서 사용하지 않음).
    - 생육조사는 정식일 직후가 아니라 평균 40~50일 이후부터 시작되며,
      수확 시작 전(pre-harvest) 구간만 봐도 이미 92%가 화방(꽃대)이
      나온 '개화 이후' 상태였음. 즉 순수 '영양생장기(개화 전)'만의
      데이터는 매우 적어 별도 모델로 분리하기엔 부적합함.
      -> 생육단계를 3단계로 쪼개지 않고, '수확 전(생육기)' 하나의
         모델 안에서 화방착과여부/정식후경과일을 피처로 사용해
         개화 전후 차이를 자동으로 학습하게 한다.
    - 27개 농가 중 26개는 생육조사가 첫 출하일보다 먼저 시작됨
      (평균 약 53일 앞섬) -> 수확 전 생육 궤적 데이터 확보 가능.

출력 컬럼 요약:
    - 농가ID, 조사일자, 이전조사일자, 경과일수
    - 환경 rolling 7일 피처 (온도/습도/CO2/일사량 등)
    - 재식밀도, 식부면적, 정식후경과일
    - 화방착과여부(그 시점 기준), 생육조사_개체수
    - 성장률 타겟: 초장_성장률, 엽수_성장률, 엽장_성장률, 엽폭_성장률,
                   엽병장_성장률, 관부직경_성장률
      (모두 "그 기간 동안의 변화량 / 경과일수" = 일평균 성장 속도)

주의:
    - 환경 원본(시간별) 데이터 로딩 부분은 01b_prepare_data_combined.py의
      load_year/build_daily_env/build_rolling_env 로직을 그대로 재사용한다.
      DATA_DIR 등 경로는 본인 환경에 맞게 01b와 동일하게 맞춰야 한다.
    - 재배정보(정식일/재식밀도/식부면적)는 농가당 1건으로 가정한다
      (01b와 동일한 방식). 만약 실제로 한 농가가 여러 재배 사이클을
      거쳤다면(작기 구분이 필요한 경우) 이 가정이 깨질 수 있으니,
      데이터 갱신 시 재확인 필요.
"""

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# ============================================
# 경로 설정 (01b_prepare_data_combined.py와 동일하게 맞출 것)
# ============================================

DATA_DIR = (
    "data/농촌진흥청_스마트팜 현장 농가 데이터_20260415"
)

GROWTH_FILES = {
    2022: 'data/딸기_2022_생육_통합.csv',
    2023: 'data/딸기_2023_생육_통합.csv',
}

TRAIN_TABLE_PATH = 'data/train_table_combined.csv'  # 첫 출하일 기준용
OUTPUT_PATH = 'train_table_growth_rate_full.csv'

env_cols = [
    '온도_외부',
    '일사량_외부',
    '누적일사량_외부',
    '온도_내부',
    '상대습도_내부',
    '잔존 이산화탄소(CO2)'
]

GROWTH_NUMERIC_COLS = ['초장', '엽수', '엽장', '엽폭', '엽병장', '관부직경']
FRUIT_COL = '화방별착과수'

EXCLUDE_2022_FARMS = [26, 27, 28, 29, 30, 31, 32, 33]  # 01b와 동일


# ============================================
# 1) 환경 데이터 로드/집계 (01b와 동일 로직 재사용)
# ============================================

def load_env(year):
    env = pd.read_csv(
        f'{DATA_DIR}/{year}/1.환경/{year}_환경_통합v1.csv',
        encoding='cp949',
        low_memory=False
    )
    straw_env = env[env['품목'] == '딸기'].copy()
    straw_env = straw_env.rename(
        columns={'잔존이산화탄소(CO2)': '잔존 이산화탄소(CO2)'}
    )

    if year == 2022:
        straw_env = straw_env[~straw_env['농가명'].isin(EXCLUDE_2022_FARMS)]

    for c in env_cols:
        straw_env[c] = pd.to_numeric(straw_env[c], errors='coerce')

    straw_env['측정시간'] = pd.to_datetime(straw_env['측정시간'])
    straw_env['날짜'] = straw_env['측정시간'].dt.date
    straw_env['농가ID'] = f'{year}_' + straw_env['농가명'].astype(str)

    return straw_env


def load_cult(year):
    cult = pd.read_csv(
        f'{DATA_DIR}/{year}/4.재배정보/{year}_재배정보_통합_VF.csv',
        encoding='cp949'
    )
    straw_cult = cult[cult['품목'] == '딸기'].copy()

    if year == 2022:
        straw_cult = straw_cult[~straw_cult['농가명'].isin(EXCLUDE_2022_FARMS)]

    straw_cult['정식일'] = pd.to_datetime(straw_cult['정식일'])
    straw_cult['농가ID'] = f'{year}_' + straw_cult['농가명'].astype(str)

    return straw_cult[['농가ID', '정식일', '재식밀도', '식부면적']].drop_duplicates('농가ID')


def build_daily_env(straw_env):
    daily_env = (
        straw_env
        .groupby(['농가ID', '날짜'])[env_cols]
        .agg(['mean', 'min', 'max'])
    )
    daily_env.columns = ['_'.join(c) for c in daily_env.columns]
    daily_env = daily_env.reset_index()
    daily_env['날짜'] = pd.to_datetime(daily_env['날짜'])
    return daily_env.sort_values(['농가ID', '날짜'])


def build_rolling_env(daily_env, window=7):
    rows = []
    for farm_id, grp in daily_env.groupby('농가ID'):
        grp = grp.set_index('날짜').sort_index()
        numeric_cols = grp.select_dtypes(include=[np.number]).columns
        rolled = grp[numeric_cols].rolling(f'{window}D', min_periods=3).mean()
        rolled = rolled.reset_index()
        rolled['농가ID'] = farm_id
        rows.append(rolled)

    rolling_env = pd.concat(rows, ignore_index=True)
    rolling_env = rolling_env.rename(columns={'날짜': '기준일'})
    return rolling_env


print("환경/재배정보 로드 중...")

env_all, cult_all = [], []
for year in (2022, 2023):
    env_all.append(load_env(year))
    cult_all.append(load_cult(year))

daily_env_all = pd.concat(
    [build_daily_env(e) for e in env_all],
    ignore_index=True
)
rolling_env_all = build_rolling_env(daily_env_all, window=7)
cult_all = pd.concat(cult_all, ignore_index=True)

print(f"환경 rolling 테이블: {rolling_env_all.shape}")


# ============================================
# 2) 생육 데이터 로드 + 정리 (본주만)
# ============================================

def load_and_clean_growth(year, path):
    g = pd.read_csv(path, encoding='cp949')
    g = g[g['품목'] == '딸기'].copy()

    g['액아구분'] = g['액아구분'].astype(str).str.strip()
    g = g[g['액아구분'].str.startswith('본주')].copy()

    g['조사일자'] = pd.to_datetime(g['조사일자'])

    for c in GROWTH_NUMERIC_COLS + [FRUIT_COL]:
        g[c] = pd.to_numeric(g[c], errors='coerce')

    g['농가ID'] = f'{year}_' + g['농가명'].astype(str)

    return g


print("\n생육 데이터 로드 중...")

growth_raw = pd.concat(
    [load_and_clean_growth(y, p) for y, p in GROWTH_FILES.items()],
    ignore_index=True
)

# 개체별 개화여부 (화방번호가 기록됐으면 개화 이후로 간주)
growth_raw['개화여부'] = growth_raw['화방번호'].notna()


# ============================================
# 3) 개체 단위 -> 농가·조사일 단위로 집계
# ============================================

fruit_per_plant = (
    growth_raw
    .groupby(['농가ID', '조사일자', '개체번호'])[FRUIT_COL]
    .sum()
    .reset_index()
)

fruit_daily = (
    fruit_per_plant
    .groupby(['농가ID', '조사일자'])[FRUIT_COL]
    .mean()
    .reset_index()
    .rename(columns={FRUIT_COL: '화방별착과수_개체평균'})
)

growth_daily = (
    growth_raw
    .groupby(['농가ID', '조사일자'])[GROWTH_NUMERIC_COLS]
    .mean()
    .reset_index()
)

n_plants = (
    growth_raw
    .groupby(['농가ID', '조사일자'])['개체번호']
    .nunique()
    .reset_index()
    .rename(columns={'개체번호': '생육조사_개체수'})
)

# 화방착과여부: 그 조사일에 한 개체라도 개화했으면 True
flowering_daily = (
    growth_raw
    .groupby(['농가ID', '조사일자'])['개화여부']
    .any()
    .reset_index()
    .rename(columns={'개화여부': '화방착과여부'})
)

growth_daily = (
    growth_daily
    .merge(fruit_daily, on=['농가ID', '조사일자'], how='left')
    .merge(n_plants, on=['농가ID', '조사일자'], how='left')
    .merge(flowering_daily, on=['농가ID', '조사일자'], how='left')
)

growth_daily = growth_daily.sort_values(['농가ID', '조사일자']).reset_index(drop=True)

print(f"농가·조사일 단위 생육 집계: {growth_daily.shape}")


# ============================================
# 4) '수확중여부' 플래그 추가 (필터링은 하지 않음)
#    - 딸기는 수확이 시작된 이후에도 계속 생장하므로(새 잎/화방 발생,
#      관부 비대 등), pre-harvest만 쓰면 데이터가 너무 적어짐(190건)
#    - 대신 전체 기간을 다 쓰되, '지금이 수확기인가 아닌가'를
#      피처(수확중여부)로 남겨서, 생육단계 추천 로직에서 참고할 수 있게 함
#      (모델을 나누지 않고, 하나의 '환경->성장률' 모델이 이 플래그와
#       정식후경과일/화방착과여부를 같이 보고 자동으로 구간별 차이를 학습)
# ============================================

train_sales = pd.read_csv(TRAIN_TABLE_PATH)
train_sales['출하일자'] = pd.to_datetime(train_sales['출하일자'])

first_sale = (
    train_sales
    .groupby('농가ID')['출하일자']
    .min()
    .rename('첫출하일')
    .reset_index()
)

growth_daily = growth_daily.merge(first_sale, on='농가ID', how='left')
growth_daily['수확중여부'] = growth_daily['조사일자'] >= growth_daily['첫출하일']

print(
    f"전체 생육조사 {len(growth_daily)}건 유지 "
    f"(수확 전 {(~growth_daily['수확중여부']).sum()}건 / "
    f"수확중 {growth_daily['수확중여부'].sum()}건)"
)


# ============================================
# 5) 조사일 간 성장률(성장 속도) 계산
#    - 농가별로 조사일자 순 정렬 후, 직전 조사값 대비 변화량 / 경과일수
# ============================================

growth_daily = growth_daily.sort_values(['농가ID', '조사일자'])

rate_rows = []

for farm_id, grp in growth_daily.groupby('농가ID'):

    grp = grp.sort_values('조사일자').reset_index(drop=True)

    grp['이전조사일자'] = grp['조사일자'].shift(1)
    grp['경과일수'] = (grp['조사일자'] - grp['이전조사일자']).dt.days

    for col in GROWTH_NUMERIC_COLS:
        prev = grp[col].shift(1)
        delta = grp[col] - prev
        grp[f'{col}_성장률'] = delta / grp['경과일수']

    rate_rows.append(grp)

growth_rate_df = pd.concat(rate_rows, ignore_index=True)

# 첫 조사(직전 값 없음)는 성장률 계산 불가 -> 제거
before_n = len(growth_rate_df)
growth_rate_df = growth_rate_df.dropna(subset=['경과일수'])
growth_rate_df = growth_rate_df[growth_rate_df['경과일수'] > 0]
print(
    f"첫 조사(직전 비교 불가) 제거: "
    f"{before_n} -> {len(growth_rate_df)}건"
)


# ============================================
# 6) 재배정보(재식밀도/식부면적/정식일) + 환경(rolling 7일) 병합
# ============================================

growth_rate_df = growth_rate_df.merge(cult_all, on='농가ID', how='left')

growth_rate_df['정식후경과일'] = (
    growth_rate_df['조사일자'] - growth_rate_df['정식일']
).dt.days

# 환경 rolling 값은 '조사일자' 기준으로 병합
# (build_rolling_env의 결과 컬럼명이 '기준일'이므로 조사일자와 맞춰 병합)
final = growth_rate_df.merge(
    rolling_env_all,
    left_on=['농가ID', '조사일자'],
    right_on=['농가ID', '기준일'],
    how='left'
)

final = final.drop(columns=['기준일'])

# ============================================
# 7) 결측 확인 및 저장
# ============================================

print(f"\n최종 생육기 학습 테이블: {final.shape}")

env_feat_cols = [
    c for c in final.columns
    if any(c.startswith(p) for p in env_cols)
]

check_cols = env_feat_cols + [c + '_성장률' for c in GROWTH_NUMERIC_COLS]
print("주요 컬럼 결측 개수:")
print(final[check_cols].isna().sum())

final.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
print(f"\n저장 완료: {OUTPUT_PATH}")
print(final.head())
