# -*- coding: utf-8 -*-
"""
AgroMind - 생육 데이터(초장·엽수 등) 병합 스크립트

목적:
    기존 train_table_combined.csv(환경+재배+판매 데이터)에
    농가별 생육 조사 데이터(초장, 엽수, 엽장, 엽폭, 엽병장, 관부직경,
    화방별착과수)를 추가하여, "환경 -> 생육 -> 생산성" 경로를
    분석할 수 있는 확장 테이블을 만든다.

생육 데이터 특징 (딸기_2022_생육_통합.csv / 딸기_2023_생육_통합.csv):
    - 개체(포기) 단위로 기록되며, 한 조사일에 농가당 여러 개체를 측정
    - 조사 주기는 매일이 아니라 약 1주 간격
    - '액아구분' 컬럼에 본주/액아(곁가지)가 섞여 있고 표기가 일관되지 않음
      (예: '본주', '본주2', '본주 '(공백), '액아', '액아-1', '액아1' 등)
    - 본 연구의 관심사는 "본체(본주)"의 생육 상태이므로 액아는 제외

병합 방식:
    - 생육 데이터는 주 단위라서 환경데이터(일 단위)처럼 날짜가 정확히
      맞아떨어지지 않음. 따라서 각 출하 건(기준일) 시점에서
      "그 시점까지의 가장 최근 생육 조사값"을 merge_asof(backward)로
      붙인다 (환경데이터의 '직전 7일 rolling'과 같은 논리:
      최근 생육 상태를 그 시점의 대표값으로 사용).
    - 생육조사일과 기준일 사이의 간격(생육조사경과일)도 함께 남겨서,
      너무 오래된 생육값이 붙는 경우를 나중에 걸러낼 수 있게 한다.
"""

import pandas as pd
import numpy as np

# ============================================
# 경로 설정 (환경에 맞게 수정)
# ============================================

GROWTH_FILES = {
    2022: 'data/딸기_2022_생육_통합.csv',
    2023: 'data/딸기_2023_생육_통합.csv',
}

TRAIN_TABLE_PATH = 'data/train_table_combined.csv'
OUTPUT_PATH = 'train_table_with_growth.csv'

# 병합에 쓸 생육 수치 컬럼
GROWTH_NUMERIC_COLS = [
    '초장',      # 초장 (키)
    '엽수',      # 엽수 (잎 개수)
    '엽장',      # 엽장
    '엽폭',      # 엽폭
    '엽병장',    # 엽병장
    '관부직경',  # 관부직경
]

# 화방별착과수는 화방번호별로 기록되어 있어 별도 집계 (개체당 총 착과수)
FRUIT_COL = '화방별착과수'

# 생육값이 기준일보다 이 일수 이상 오래됐으면 신뢰도가 떨어진다고 볼 수 있는 기준
# (분석 시 참고용 플래그로만 사용, 여기서는 컬럼만 만들어 둠)
STALE_THRESHOLD_DAYS = 14


def load_and_clean_growth(year, path):
    """생육 데이터 로드 + 본주만 필터링 + 날짜 처리"""

    g = pd.read_csv(path, encoding='cp949')

    # 딸기만 (원본이 이미 딸기만 있는 경우가 많지만 안전하게 필터링)
    g = g[g['품목'] == '딸기'].copy()

    # 액아구분 표기 정리: 공백 제거 후 '본주'로 시작하는 것만 사용
    # (본주, 본주2, '본주 ' 등은 모두 본체로 간주 / 액아·액아1·액1 등은 제외)
    g['액아구분'] = g['액아구분'].astype(str).str.strip()
    g = g[g['액아구분'].str.startswith('본주')].copy()

    # 날짜 처리
    g['조사일자'] = pd.to_datetime(g['조사일자'])

    # 수치 컬럼 강제 형변환 (문자 섞여 있을 가능성 대비)
    for c in GROWTH_NUMERIC_COLS + [FRUIT_COL]:
        g[c] = pd.to_numeric(g[c], errors='coerce')

    # 학습 테이블과 동일한 농가ID 포맷: '연도_농가명'
    g['농가ID'] = f'{year}_' + g['농가명'].astype(str)

    return g


def aggregate_growth_daily(g):
    """개체 단위 -> 농가·조사일 단위로 집계"""

    # 화방별착과수는 화방(꽃대) 단위 기록이므로,
    # 먼저 개체별로 합산(그 개체의 총 착과수)한 뒤 농가 평균을 낸다.
    fruit_per_plant = (
        g
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

    # 나머지 생육 수치는 개체 평균으로 집계
    growth_daily = (
        g
        .groupby(['농가ID', '조사일자'])[GROWTH_NUMERIC_COLS]
        .mean()
        .reset_index()
    )

    # 조사에 사용된 개체 수도 함께 기록 (표본 크기 참고용)
    n_plants = (
        g
        .groupby(['농가ID', '조사일자'])['개체번호']
        .nunique()
        .reset_index()
        .rename(columns={'개체번호': '생육조사_개체수'})
    )

    growth_daily = growth_daily.merge(
        fruit_daily, on=['농가ID', '조사일자'], how='left'
    )
    growth_daily = growth_daily.merge(
        n_plants, on=['농가ID', '조사일자'], how='left'
    )

    growth_daily = growth_daily.sort_values(['농가ID', '조사일자'])

    return growth_daily


# ============================================
# 1) 생육 데이터 로드 + 집계 (2022 + 2023 통합)
# ============================================

growth_all = []

for year, path in GROWTH_FILES.items():
    g = load_and_clean_growth(year, path)
    g_daily = aggregate_growth_daily(g)
    growth_all.append(g_daily)

    print(
        f"[{year}] 생육 원본 {len(g):,}행(개체×조사일, 본주만) "
        f"-> 농가·조사일 집계 {len(g_daily):,}행 "
        f"(농가 {g_daily['농가ID'].nunique()}개)"
    )

growth_daily_all = pd.concat(growth_all, ignore_index=True)
growth_daily_all = growth_daily_all.sort_values(['농가ID', '조사일자'])

# ============================================
# 2) 기존 학습 테이블 로드
# ============================================

train = pd.read_csv(TRAIN_TABLE_PATH)
train['출하일자'] = pd.to_datetime(train['출하일자'])
train['기준일'] = pd.to_datetime(train['기준일'])

print(f"\n기존 학습 테이블: {train.shape}")

# ============================================
# 3) merge_asof로 "가장 최근 생육 조사값" 병합
#    - 농가(그룹)별로 기준일 <= 조사일자가 아니라
#      "기준일 시점까지 관측된 가장 최근 생육값"을 붙여야 하므로
#      direction='backward' 사용 (조사일자 <= 기준일 중 가장 가까운 값)
# ============================================

merged_parts = []

for farm_id, farm_df in train.groupby('농가ID'):

    farm_df = farm_df.sort_values('기준일')

    g_sub = growth_daily_all[
        growth_daily_all['농가ID'] == farm_id
    ].sort_values('조사일자')

    if len(g_sub) == 0:
        # 해당 농가의 생육 데이터가 아예 없는 경우
        # (현재 데이터셋에서는 발생하지 않지만 안전장치로 둠)
        merged = farm_df.copy()
        for c in GROWTH_NUMERIC_COLS + ['화방별착과수_개체평균', '생육조사_개체수']:
            merged[c] = np.nan
        merged['생육조사일자'] = pd.NaT
    else:
        merged = pd.merge_asof(
            farm_df,
            g_sub,
            left_on='기준일',
            right_on='조사일자',
            by='농가ID',
            direction='backward'
        )
        merged = merged.rename(columns={'조사일자': '생육조사일자'})

    merged_parts.append(merged)

train_with_growth = pd.concat(merged_parts, ignore_index=True)

# 생육조사경과일: 기준일과 생육조사일자 사이 간격 (참고/필터링용)
train_with_growth['생육조사경과일'] = (
    train_with_growth['기준일'] - train_with_growth['생육조사일자']
).dt.days

# ============================================
# 4) 결과 확인 및 저장
# ============================================

n_missing = train_with_growth['생육조사일자'].isna().sum()
n_stale = (
    train_with_growth['생육조사경과일'] > STALE_THRESHOLD_DAYS
).sum()

print(f"\n병합 결과: {train_with_growth.shape}")
print(f"생육값이 아예 안 붙은 행 (해당 시점 이전 조사기록 없음): {n_missing}건")
print(
    f"생육조사경과일 > {STALE_THRESHOLD_DAYS}일인 행 "
    f"(오래된 생육값이 붙은 경우, 참고용 플래그): {n_stale}건"
)

print("\n생육조사경과일 분포:")
print(train_with_growth['생육조사경과일'].describe())

train_with_growth.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding='utf-8-sig'
)

print(f"\n저장 완료: {OUTPUT_PATH}")
print(train_with_growth.head())
