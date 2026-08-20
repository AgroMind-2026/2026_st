# -*- coding: utf-8 -*-
"""
AgroMind - 데이터 전처리 및 병합 스크립트 (2022+2023 통합)
목표: 환경 데이터(시간별) + 재배정보 + 판매 데이터를 결합하여
      "환경조건 -> 단위면적당 출하량" 예측을 위한 학습용 테이블 생성 (딸기)
"""
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

env_cols = ['온도_외부', '일사량_외부', '누적일사량_외부', '온도_내부',
            '상대습도_내부', '잔존 이산화탄소(CO2)']

# 2022년 재식밀도 단위 이상 의심 농가 (154~205 range, 나머지는 3~8) -> 제외
EXCLUDE_2022_FARMS = [26, 27, 28, 29, 30, 31, 32, 33]


def load_year(year):
    """연도별 환경/판매/재배정보를 불러와 딸기만 필터링"""
    env = pd.read_csv(f'/mnt/user-data/uploads/{year}_환경_통합v1.csv', encoding='cp949', low_memory=False)
    sales = pd.read_csv(f'/mnt/user-data/uploads/{year}_판매_통합.csv', encoding='cp949')
    cult = pd.read_csv(f'/mnt/user-data/uploads/{year}_재배정보_통합_VF.csv', encoding='cp949')

    straw_env = env[env['품목'] == '딸기'].copy()
    straw_sales = sales[sales['품목'] == '딸기'].copy()
    straw_cult = cult[cult['품목'] == '딸기'].copy()

    # 연도별 컬럼명 표기 차이 표준화 (예: '잔존이산화탄소(CO2)' vs '잔존 이산화탄소(CO2)')
    straw_env = straw_env.rename(columns={'잔존이산화탄소(CO2)': '잔존 이산화탄소(CO2)'})

    if year == 2022:
        straw_env = straw_env[~straw_env['농가명'].isin(EXCLUDE_2022_FARMS)]
        straw_sales = straw_sales[~straw_sales['농가명'].isin(EXCLUDE_2022_FARMS)]
        straw_cult = straw_cult[~straw_cult['농가명'].isin(EXCLUDE_2022_FARMS)]

    for c in env_cols:
        straw_env[c] = pd.to_numeric(straw_env[c], errors='coerce')

    straw_env['측정시간'] = pd.to_datetime(straw_env['측정시간'])
    straw_env['날짜'] = straw_env['측정시간'].dt.date
    straw_sales['출하일자'] = pd.to_datetime(straw_sales['출하일자'])
    straw_cult['정식일'] = pd.to_datetime(straw_cult['정식일'])

    # 연도 접두어를 붙여 농가 ID 충돌 방지 (2022-3번 농가 != 2023-3번 농가)
    for d in (straw_env, straw_sales, straw_cult):
        d['농가ID'] = f'{year}_' + d['농가명'].astype(str)

    return straw_env, straw_sales, straw_cult


def build_daily_env(straw_env):
    daily_env = (
        straw_env.groupby(['농가ID', '날짜'])[env_cols]
        .agg(['mean', 'min', 'max'])
    )
    daily_env.columns = ['_'.join(c) for c in daily_env.columns]
    daily_env = daily_env.reset_index()
    daily_env['날짜'] = pd.to_datetime(daily_env['날짜'])
    daily_env = daily_env.sort_values(['농가ID', '날짜'])
    return daily_env


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


all_tables = []

for year in (2022, 2023):
    straw_env, straw_sales, straw_cult = load_year(year)
    print(f"[{year}] 환경 {len(straw_env):,} / 판매 {len(straw_sales):,} / 농가 {straw_cult['농가ID'].nunique()}")

    daily_env = build_daily_env(straw_env)
    rolling_env = build_rolling_env(daily_env, window=7)

    straw_sales['기준일'] = straw_sales['출하일자'] - pd.Timedelta(days=1)
    merged = pd.merge(straw_sales, rolling_env, on=['농가ID', '기준일'], how='inner')

    cult_small = straw_cult[['농가ID', '정식일', '재식밀도', '식부면적']].drop_duplicates('농가ID')
    merged = pd.merge(merged, cult_small, on='농가ID', how='left')
    merged['정식후경과일'] = (merged['출하일자'] - merged['정식일']).dt.days
    merged = merged[merged['정식후경과일'] >= 0]
    merged['연도'] = year

    env_feat_cols = [c for c in merged.columns if any(c.startswith(p) for p in env_cols)]
    group_keys = ['농가ID', '연도', '출하일자', '기준일', '정식후경과일', '재식밀도', '식부면적'] + env_feat_cols
    agg_daily = merged.groupby(group_keys, as_index=False)[['총출하량', '판매금액']].sum()

    all_tables.append(agg_daily)

final = pd.concat(all_tables, ignore_index=True)
print(f"\n2022+2023 통합 학습 테이블: {final.shape}")
print(f"고유 농가 수: {final['농가ID'].nunique()}")
print(final.isna().sum().sum(), "개 결측치 (총 셀)")

final = final.dropna()
final.to_csv('/home/claude/agromind/train_table_combined.csv', index=False, encoding='utf-8-sig')
print("저장 완료: train_table_combined.csv")
print(final.head())
