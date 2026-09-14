# -*- coding: utf-8 -*-
"""
AgroMind - 환경 추천 엔진 (생육 성장률 모델 기반)

목적:
    11_train_growth_rate_model.py로 학습/저장한 지표별 RandomForest
    모델(rf_*.joblib)을 불러와서, "현재 환경"과 "생육 단계"를 입력받아
    "성장에 더 도움이 되는 환경 조합"을 탐색하고, 그 차이를
    구체적인 조치(예: 내부온도 2도 낮추기)로 변환해 출력한다.

핵심 설계 결정 (지난 대화 반영):
    1. 지표(초장/엽수/엽장/엽폭/엽병장/관부직경)마다 모델 신뢰도(R2)가
       크게 다름 (엽폭 0.17, 엽장 0.14 vs 초장/엽수/관부직경은 0 이하).
       -> R2가 낮은(0 이하) 지표는 추천 근거에서 사실상 제외되도록,
          "신뢰도 가중치 = max(R2, 0)"으로 정규화해서 종합 점수를 만든다.
          (즉 못 맞추는 지표가 추천 방향을 왜곡하지 않게 함)
    2. 사용자가 직접 통제 가능한 변수만 탐색 대상으로 삼는다.
       - 통제 가능: 내부온도, 내부습도, CO2 (냉난방/환기/CO2 발생기로 조절)
       - 통제 불가능(그대로 유지): 외부온도/일사량/누적일사량 (날씨),
         재식밀도/식부면적(이미 정해진 값), 정식후경과일/화방착과여부/
         수확중여부(생육 진행 상태, 사용자가 입력)
    3. mean을 조절할 때 min/max도 같은 폭으로 같이 이동시켜
       (하루 중 변동 폭 자체는 유지) 비현실적인 조합(min>max 등)을 방지한다.

사용 전 준비:
    - growth_models/rf_*.joblib (11번 스크립트 실행 결과)
    - growth_model_cv_summary.csv (지표별 Pooled R2, 신뢰도 가중치 계산용)
    - data/train_table_growth_rate_full.csv (탐색 범위 설정을 위한
      환경변수 분포 참고용)
"""

import pandas as pd
import numpy as np
import joblib
import os
import itertools

MODEL_DIR = "growth_models"
CV_SUMMARY_PATH = "growth_model_cv_summary.csv"
REF_DATA_PATH = "data/train_table_growth_rate_full.csv"

TARGETS = [
    '초장_성장률', '엽수_성장률', '엽장_성장률',
    '엽폭_성장률', '엽병장_성장률', '관부직경_성장률',
]

# 모델이 기대하는 피처 (11번 스크립트와 동일해야 함)
ENV_PREFIXES = [
    '온도_외부', '일사량_외부', '누적일사량_외부',
    '온도_내부', '상대습도_내부', '잔존 이산화탄소(CO2)'
]

BASE_FEATS = [
    '재식밀도', '식부면적', '정식후경과일',
    '화방착과여부', '수확중여부', '화방별착과수_개체평균', '경과일수',
]

# 탐색(조절) 대상 - 실제 농가에서 제어 가능한 변수만
CONTROLLABLE_MEAN_VARS = [
    '온도_내부_mean',
    '상대습도_내부_mean',
    '잔존 이산화탄소(CO2)_mean',
]

# 변수별 탐색 범위(현재값 대비 +-) 및 스텝
SEARCH_RANGE = {
    '온도_내부_mean': {'delta_range': np.arange(-4, 4.01, 1.0)},        # 도(C)
    '상대습도_내부_mean': {'delta_range': np.arange(-15, 15.01, 5.0)},   # %
    '잔존 이산화탄소(CO2)_mean': {'delta_range': np.arange(-200, 200.01, 50.0)},  # ppm
}


# ============================================
# 1) 모델 및 신뢰도 가중치 로드
# ============================================

def load_models_and_weights():
    models = {}
    for t in TARGETS:
        path = os.path.join(MODEL_DIR, f"rf_{t}.joblib")
        if os.path.exists(path):
            models[t] = joblib.load(path)
        else:
            print(f"⚠ 모델 없음: {path} (해당 지표는 추천에서 제외)")

    weights = {t: 0.0 for t in models}

    if os.path.exists(CV_SUMMARY_PATH):
        cv = pd.read_csv(CV_SUMMARY_PATH)
        rf_cv = cv[cv['model'] == 'RandomForest']
        for _, row in rf_cv.iterrows():
            t = row['target']
            if t in weights:
                # 신뢰도 가중치: R2가 0 이하인 지표는 가중치 0
                weights[t] = max(row['Pooled_R2'], 0.0)
    else:
        print(
            f"⚠ {CV_SUMMARY_PATH} 없음 -> 모든 지표를 동일 가중치로 사용 "
            f"(추천드리지 않음, 11번 스크립트를 먼저 실행하세요)"
        )
        weights = {t: 1.0 for t in models}

    total = sum(weights.values())
    if total <= 0:
        raise ValueError(
            "모든 지표의 신뢰도(R2)가 0 이하입니다. "
            "지금 모델로는 신뢰할 만한 추천을 만들 수 없습니다. "
            "생육데이터/피처를 보강한 뒤 다시 시도하세요."
        )

    weights = {t: w / total for t, w in weights.items()}

    print("지표별 신뢰도 가중치 (정규화됨):")
    for t, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {t}: {w:.3f}")

    return models, weights


def get_feature_cols(ref_df):
    env_feat_cols = [
        c for c in ref_df.columns
        if any(c.startswith(p) for p in ENV_PREFIXES)
    ]
    return env_feat_cols + BASE_FEATS


# ============================================
# 2) 종합 성장 점수 계산
#    - 신뢰도 가중치로 표준화된 지표들을 합쳐 "종합 성장 점수" 산출
#    - 표준화 기준(평균/표준편차)은 참고 데이터(REF_DATA_PATH)에서 계산
# ============================================

def compute_target_stats(ref_df):
    stats = {}
    for t in TARGETS:
        if t in ref_df.columns:
            stats[t] = {
                'mean': ref_df[t].mean(),
                'std': ref_df[t].std() if ref_df[t].std() > 1e-8 else 1.0,
            }
    return stats


def composite_score(pred_dict, weights, target_stats):
    """지표별 예측 성장률을 z-score로 표준화한 뒤 가중합"""
    score = 0.0
    for t, pred in pred_dict.items():
        if t not in weights or weights[t] == 0:
            continue
        z = (pred - target_stats[t]['mean']) / target_stats[t]['std']
        score += weights[t] * z
    return score


# ============================================
# 3) 현재 상태 -> 모델 입력 벡터 구성
# ============================================

def build_feature_vector(current_state, feature_cols):
    """
    current_state: dict. 최소한 다음 키를 포함해야 함:
        - 온도_외부_mean/min/max, 일사량_외부_mean/min/max,
          누적일사량_외부_mean/min/max, 온도_내부_mean/min/max,
          상대습도_내부_mean/min/max, 잔존 이산화탄소(CO2)_mean/min/max
        - 재식밀도, 식부면적, 정식후경과일, 화방착과여부(0/1),
          수확중여부(0/1), 화방별착과수_개체평균
    경과일수는 항상 7(표준 조사 간격)로 고정해서 넣는다.
    """
    row = dict(current_state)
    row['경과일수'] = 7  # 추론 시 표준값 고정 (사용자가 조절 불가한 변수)

    missing = [c for c in feature_cols if c not in row]
    if missing:
        raise ValueError(f"current_state에 다음 값이 없습니다: {missing}")

    return pd.DataFrame([{c: row[c] for c in feature_cols}])


def predict_all_targets(models, X):
    return {t: float(m.predict(X)[0]) for t, m in models.items()}


# ============================================
# 4) 그리드 탐색으로 최적 조합 찾기
# ============================================

def search_best_environment(current_state, models, weights, target_stats, feature_cols):

    base_X = build_feature_vector(current_state, feature_cols)
    base_pred = predict_all_targets(models, base_X)
    base_score = composite_score(base_pred, weights, target_stats)

    # 각 통제가능 변수의 델타 후보 조합 (전부 조합 -> grid search)
    delta_lists = [
        SEARCH_RANGE[v]['delta_range'] for v in CONTROLLABLE_MEAN_VARS
    ]

    best_score = base_score
    best_deltas = {v: 0.0 for v in CONTROLLABLE_MEAN_VARS}
    best_pred = base_pred

    for deltas in itertools.product(*delta_lists):

        candidate = dict(current_state)

        for var, delta in zip(CONTROLLABLE_MEAN_VARS, deltas):
            prefix = var.replace('_mean', '')
            # mean/min/max를 같은 폭으로 이동 (하루 중 변동 폭 유지)
            for suffix in ['_mean', '_min', '_max']:
                key = f'{prefix}{suffix}'
                if key in candidate:
                    candidate[key] = candidate[key] + delta

        X = build_feature_vector(candidate, feature_cols)
        pred = predict_all_targets(models, X)
        score = composite_score(pred, weights, target_stats)

        if score > best_score:
            best_score = score
            best_deltas = dict(zip(CONTROLLABLE_MEAN_VARS, deltas))
            best_pred = pred

    return {
        'base_score': base_score,
        'base_pred': base_pred,
        'best_score': best_score,
        'best_pred': best_pred,
        'best_deltas': best_deltas,
        'improvement': best_score - base_score,
    }


# ============================================
# 5) 결과를 사람이 읽을 수 있는 추천 문구로 변환
# ============================================

VAR_LABEL = {
    '온도_내부_mean': '내부 온도',
    '상대습도_내부_mean': '내부 습도',
    '잔존 이산화탄소(CO2)_mean': 'CO2 농도',
}
VAR_UNIT = {
    '온도_내부_mean': '℃',
    '상대습도_내부_mean': '%',
    '잔존 이산화탄소(CO2)_mean': 'ppm',
}


def format_recommendation(result):
    lines = []

    if result['improvement'] <= 1e-6:
        lines.append("현재 환경이 이미 양호합니다. 큰 조정이 필요하지 않습니다.")
        return "\n".join(lines)

    lines.append("=== 환경 조절 추천 ===")
    for var, delta in result['best_deltas'].items():
        if abs(delta) < 1e-6:
            continue
        label = VAR_LABEL.get(var, var)
        unit = VAR_UNIT.get(var, '')
        direction = "높이세요" if delta > 0 else "낮추세요"
        lines.append(f" - {label}: 현재 대비 {abs(delta):.1f}{unit} {direction}")

    lines.append(
        f"\n종합 성장 점수(표준화): {result['base_score']:.3f} -> "
        f"{result['best_score']:.3f} "
        f"(개선폭 {result['improvement']:.3f})"
    )

    return "\n".join(lines)


# ============================================
# 사용 예시
# ============================================

if __name__ == "__main__":

    models, weights = load_models_and_weights()

    ref_df = pd.read_csv(REF_DATA_PATH)
    feature_cols = get_feature_cols(ref_df)
    target_stats = compute_target_stats(ref_df)

    # 예시: 사용자가 입력했다고 가정한 현재 상태
    # (실제 서비스에서는 대시보드 입력값으로 대체)
    example_state = {
        '온도_외부_mean': 12.0, '온도_외부_min': 5.0, '온도_외부_max': 20.0,
        '일사량_외부_mean': 300.0, '일사량_외부_min': 0.0, '일사량_외부_max': 700.0,
        '누적일사량_외부_mean': 5000.0, '누적일사량_외부_min': 0.0, '누적일사량_외부_max': 9000.0,
        '온도_내부_mean': 28.0, '온도_내부_min': 20.0, '온도_내부_max': 33.0,
        '상대습도_내부_mean': 42.0, '상대습도_내부_min': 30.0, '상대습도_내부_max': 55.0,
        '잔존 이산화탄소(CO2)_mean': 380.0, '잔존 이산화탄소(CO2)_min': 300.0, '잔존 이산화탄소(CO2)_max': 450.0,
        '재식밀도': 5.5,
        '식부면적': 1000.0,
        '정식후경과일': 90,
        '화방착과여부': 1,
        '수확중여부': 0,
        '화방별착과수_개체평균': 3.0,
    }

    print("\n" + "=" * 50)
    print("입력된 현재 환경 상태로 추천 계산 중...")
    print("=" * 50)

    result = search_best_environment(
        example_state, models, weights, target_stats, feature_cols
    )

    print("\n" + format_recommendation(result))
