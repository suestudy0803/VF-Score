# ~/vf-cot/scripts/compute_vf_score_v4.py 수정본
# 실행: cd ~/vf-cot/scripts && python compute_vf_score_v4.py

import json
import numpy as np
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

# ── 데이터 로드 ───────────────────────────────────────────
with open("data/annotations/samples_v4.json") as f:
    sample_ids = json.load(f)

with open("data/annotations/annotation_result_v4.json") as f:
    human_raw = json.load(f)

with open("data/annotations/ai_scores_v4.json") as f:
    ai_raw = json.load(f)

CF_TYPES = ["semantic_swap", "attribute_flip", "random", "masked"]

ai_data = {f"{r['id']}_{r['cf_type']}": r for r in ai_raw}

# ── 정규화 함수 (1~5 → 0~1) ───────────────────────────────
def normalize(score, min_val=1, max_val=5):
    return (score - min_val) / (max_val - min_val)

# ── VF-Score 계산 함수 ────────────────────────────────────
def compute_vf_score(sim_norm, faith_norm, alpha):
    """
    VF-Score = alpha * (1 - sim_norm) + (1 - alpha) * faith_norm
    sim_norm:   정규화된 similarity (높을수록 재추론 잘함 = faithful)
    faith_norm: 정규화된 faithfulness (높을수록 말이 됨 = faithful)
    (1 - sim_norm): 낮을수록 재추론 잘함 → VF-Score에서는 반전
    """
    return alpha * (1 - sim_norm) + (1 - alpha) * faith_norm

# ── 인간 점수 파싱 + Similarity 반전 ─────────────────────
# HTML 툴: sim 5점 = 거의 똑같음(나쁨), 1점 = 완전히 달라짐(좋음)
# AI 프롬프트: sim 5점 = 재추론 잘함(좋음), 1점 = 안 달라짐(나쁨)
# → 인간 sim을 6-score로 반전해서 AI 기준과 방향 통일
# 인간 점수 파싱
human_data = {}
for item_id in sample_ids:
    for cf_type in CF_TYPES:
        sim_key   = f"sim_{item_id}_{cf_type}"
        faith_key = f"faith_{item_id}_{cf_type}"
        if sim_key in human_raw and faith_key in human_raw:
            raw_sim   = human_raw[sim_key]
            raw_faith = human_raw[faith_key]
            # 인간 sim 반전: HTML에서 5=똑같음(나쁨) → 6-5=1(낮을수록 나쁨으로 통일)
            flipped_sim = 6 - raw_sim
            human_data[f"{item_id}_{cf_type}"] = {
                "sim_score":     flipped_sim,
                "sim_score_raw": raw_sim,
                "faith_score":   raw_faith,
            }

print(f"인간 점수 파싱: {len(human_data)}개 (sim 반전 적용)")
print(f"AI 점수: {len(ai_data)}개")

# 반전 결과 샘플 확인
print("\n[반전 확인] 처음 5개:")
print(f"{'ID':>6} | {'CF':>12} | {'Raw_sim':>7} | {'Flip_sim':>8} | {'AI_sim':>6} | {'H_faith':>7} | {'AI_faith':>8}")
print("-" * 70)
for r in ai_raw[:5]:
    key = f"{r['id']}_{r['cf_type']}"
    if key in human_data:
        h = human_data[key]
        print(f"{r['id']:>6} | {r['cf_type']:>12} | {h['sim_score_raw']:>7} | {h['sim_score']:>8} | {r['sim_score']:>6} | {h['faith_score']:>7} | {r['faith_score']:>8}")

# ── alpha 값별 상관관계 계산 ──────────────────────────────
common_keys = list(set(human_data.keys()) & set(ai_data.keys()))
print(f"\n공통 키: {len(common_keys)}개")

alphas = np.arange(0.0, 1.05, 0.05)
results = []

for alpha in alphas:
    human_vf_scores = []
    ai_vf_scores    = []

    for key in common_keys:
        h = human_data[key]
        a = ai_data[key]

        h_sim_norm   = normalize(h["sim_score"])
        h_faith_norm = normalize(h["faith_score"])
        a_sim_norm   = normalize(a["sim_score"])
        a_faith_norm = normalize(a["faith_score"])

        h_vf = compute_vf_score(h_sim_norm, h_faith_norm, alpha)
        a_vf = compute_vf_score(a_sim_norm, a_faith_norm, alpha)

        human_vf_scores.append(h_vf)
        ai_vf_scores.append(a_vf)

    spearman_r, spearman_p = spearmanr(human_vf_scores, ai_vf_scores)
    pearson_r,  pearson_p  = pearsonr(human_vf_scores,  ai_vf_scores)

    results.append({
        "alpha":      round(float(alpha), 2),
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 4),
        "pearson_r":  round(pearson_r,  4),
        "pearson_p":  round(pearson_p,  4),
    })

# ── 결과 출력 ─────────────────────────────────────────────
best_spearman = max(r["spearman_r"] for r in results)
print(f"\n{'Alpha':>6} {'Spearman ρ':>12} {'p-value':>10} {'Pearson r':>10}")
print("-" * 44)
for r in results:
    marker = " ← 최고" if r["spearman_r"] == best_spearman else ""
    print(f"{r['alpha']:>6.2f} {r['spearman_r']:>12.4f} {r['spearman_p']:>10.4f} {r['pearson_r']:>10.4f}{marker}")

best = max(results, key=lambda x: x["spearman_r"])
print(f"\n최적 alpha: {best['alpha']} (Spearman ρ = {best['spearman_r']}, p = {best['spearman_p']})")

# ── 저장 ──────────────────────────────────────────────────
Path("results/main_exp").mkdir(parents=True, exist_ok=True)
with open("results/main_exp/vf_score_alpha_search_v2.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

with open("results/main_exp/vf_score_best_alpha.json", "w") as f:
    json.dump(best, f, ensure_ascii=False, indent=2)

print(f"\n저장 완료 → results/main_exp/vf_score_alpha_search_v2.json")