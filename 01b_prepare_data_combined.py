# 2022+2023 딸기 데이터 병합 스크립트
# 환경 일별집계 → 7일 rolling → 판매/재배정보 병합 → 이상치 8개 농가 제외

# -*- coding: utf-8 -*-
"""
AgroMind - 데이터 전처리 및 병합 스크립트 (2022+2023 통합)
목표: 환경 데이터(시간별) + 재배정보 + 판매 데이터를 결합하여
      "환경조건 -> 단위면적당 출하량" 예측을 위한 학습용 테이블 생성 (딸기)
"""

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# ============================================
# 로컬 데이터 경로
# ============================================

DATA_DIR = (
    "/Users/hannie/Desktop/"
    "농촌진흥청_스마트팜 현장 농가 데이터_20260611/"
    "농촌진흥청_스마트팜 현장 농가 데이터_20260415"
)

env_cols = [
    '온도_외부',
    '일사량_외부',
    '누적일사량_외부',
    '온도_내부',
    '상대습도_내부',
    '잔존 이산화탄소(CO2)'
]

# 2022년 재식밀도 단위 이상 의심 농가
# (154~205 range, 나머지는 3~8) -> 제외
EXCLUDE_2022_FARMS = [26, 27, 28, 29, 30, 31, 32, 33]


def load_year(year):
    """연도별 환경/판매/재배정보를 불러와 딸기만 필터링"""

    # 환경 데이터
    env = pd.read_csv(
        f'{DATA_DIR}/{year}/1.환경/{year}_환경_통합v1.csv',
        encoding='cp949',
        low_memory=False
    )

    # 판매 데이터
    sales = pd.read_csv(
        f'{DATA_DIR}/{year}/3.생산/{year}_판매_통합.csv',
        encoding='cp949'
    )

    # 재배정보
    cult = pd.read_csv(
        f'{DATA_DIR}/{year}/4.재배정보/{year}_재배정보_통합_VF.csv',
        encoding='cp949'
    )

    # 딸기 데이터만 필터링
    straw_env = env[env['품목'] == '딸기'].copy()
    straw_sales = sales[sales['품목'] == '딸기'].copy()
    straw_cult = cult[cult['품목'] == '딸기'].copy()

    # 연도별 컬럼명 표기 차이 표준화
    # 예: '잔존이산화탄소(CO2)' vs '잔존 이산화탄소(CO2)'
    straw_env = straw_env.rename(
        columns={
            '잔존이산화탄소(CO2)': '잔존 이산화탄소(CO2)'
        }
    )

    # 2022년 이상치 농가 제외
    if year == 2022:
        straw_env = straw_env[
            ~straw_env['농가명'].isin(EXCLUDE_2022_FARMS)
        ]

        straw_sales = straw_sales[
            ~straw_sales['농가명'].isin(EXCLUDE_2022_FARMS)
        ]

        straw_cult = straw_cult[
            ~straw_cult['농가명'].isin(EXCLUDE_2022_FARMS)
        ]

    # 환경 변수 숫자형 변환
    for c in env_cols:
        straw_env[c] = pd.to_numeric(
            straw_env[c],
            errors='coerce'
        )

    # 날짜 형식 변환
    straw_env['측정시간'] = pd.to_datetime(
        straw_env['측정시간']
    )

    straw_env['날짜'] = straw_env['측정시간'].dt.date

    straw_sales['출하일자'] = pd.to_datetime(
        straw_sales['출하일자']
    )

    straw_cult['정식일'] = pd.to_datetime(
        straw_cult['정식일']
    )

    # 연도 접두어를 붙여 농가 ID 충돌 방지
    # 예: 2022년 3번 농가 != 2023년 3번 농가
    for d in (straw_env, straw_sales, straw_cult):
        d['농가ID'] = (
            f'{year}_' + d['농가명'].astype(str)
        )

    return straw_env, straw_sales, straw_cult


def build_daily_env(straw_env):
    """시간별 환경 데이터를 농가별/일별 데이터로 집계"""

    daily_env = (
        straw_env
        .groupby(['농가ID', '날짜'])[env_cols]
        .agg(['mean', 'min', 'max'])
    )

    # MultiIndex 컬럼을 일반 컬럼명으로 변경
    daily_env.columns = [
        '_'.join(c)
        for c in daily_env.columns
    ]

    daily_env = daily_env.reset_index()

    daily_env['날짜'] = pd.to_datetime(
        daily_env['날짜']
    )

    daily_env = daily_env.sort_values(
        ['농가ID', '날짜']
    )

    return daily_env


def build_rolling_env(daily_env, window=7):
    """최근 7일 환경 데이터를 rolling 평균으로 생성"""

    rows = []

    for farm_id, grp in daily_env.groupby('농가ID'):

        grp = (
            grp
            .set_index('날짜')
            .sort_index()
        )

        numeric_cols = grp.select_dtypes(
            include=[np.number]
        ).columns

        rolled = (
            grp[numeric_cols]
            .rolling(
                f'{window}D',
                min_periods=3
            )
            .mean()
        )

        rolled = rolled.reset_index()

        rolled['농가ID'] = farm_id

        rows.append(rolled)

    rolling_env = pd.concat(
        rows,
        ignore_index=True
    )

    rolling_env = rolling_env.rename(
        columns={'날짜': '기준일'}
    )

    return rolling_env


# ============================================
# 2022 + 2023 데이터 처리
# ============================================

all_tables = []

for year in (2022, 2023):

    straw_env, straw_sales, straw_cult = load_year(year)

    print(
        f"[{year}] "
        f"환경 {len(straw_env):,} / "
        f"판매 {len(straw_sales):,} / "
        f"농가 {straw_cult['농가ID'].nunique()}"
    )

    # 환경 데이터 일별 집계
    daily_env = build_daily_env(straw_env)

    # 최근 7일 rolling 환경 데이터
    rolling_env = build_rolling_env(
        daily_env,
        window=7
    )

    # 출하일 전날까지의 환경을 사용
    straw_sales['기준일'] = (
        straw_sales['출하일자']
        - pd.Timedelta(days=1)
    )

    # 판매 + 환경 병합
    merged = pd.merge(
        straw_sales,
        rolling_env,
        on=['농가ID', '기준일'],
        how='inner'
    )

    # 필요한 재배정보만 추출
    cult_small = (
        straw_cult[
            [
                '농가ID',
                '정식일',
                '재식밀도',
                '식부면적'
            ]
        ]
        .drop_duplicates('농가ID')
    )

    # 재배정보 병합
    merged = pd.merge(
        merged,
        cult_small,
        on='농가ID',
        how='left'
    )

    # 정식 후 며칠이 지났는지 계산
    merged['정식후경과일'] = (
        merged['출하일자']
        - merged['정식일']
    ).dt.days

    # 정식 이전 출하 데이터 제거
    merged = merged[
        merged['정식후경과일'] >= 0
    ]

    merged['연도'] = year

    # 환경 feature 컬럼 추출
    env_feat_cols = [
        c
        for c in merged.columns
        if any(
            c.startswith(p)
            for p in env_cols
        )
    ]

    group_keys = [
        '농가ID',
        '연도',
        '출하일자',
        '기준일',
        '정식후경과일',
        '재식밀도',
        '식부면적'
    ] + env_feat_cols

    # 하루에 판매 기록이 여러 개 있을 경우 합산
    agg_daily = (
        merged
        .groupby(
            group_keys,
            as_index=False
        )[
            ['총출하량', '판매금액']
        ]
        .sum()
    )

    all_tables.append(agg_daily)


# ============================================
# 2022 + 2023 최종 통합
# ============================================

final = pd.concat(
    all_tables,
    ignore_index=True
)

print(
    f"\n2022+2023 통합 학습 테이블: "
    f"{final.shape}"
)

print(
    f"고유 농가 수: "
    f"{final['농가ID'].nunique()}"
)

print(
    final.isna().sum().sum(),
    "개 결측치 (총 셀)"
)

# 결측치 제거
final = final.dropna()

# ============================================
# 결과 저장
# 현재 실행 중인 프로젝트 폴더에 저장됨
# ============================================

final.to_csv(
    'train_table_combined.csv',
    index=False,
    encoding='utf-8-sig'
)

print(
    "저장 완료: train_table_combined.csv"
)

print(final.head())