# ~/vf-cot/scripts/validate_vf_score_v2.py
# 실행: cd ~/vf-cot/scripts && python validate_vf_score_v2.py

import json
import numpy as np
from scipy.stats import spearmanr, pearsonr

# ── 데이터 로드 ───────────────────────────────────────────
# 반드시 samples_v4.json의 25개 ID 사용
with open("data/annotations/samples_v4.json") as f:
    sample_ids = json.load(f)
print(f"샘플 ID: {len(sample_ids)}개 (samples_v4.json)")

with open("data/annotations/annotation_result_v4.json") as f:
    human_raw = json.load(f)

# v2 AI 점수에서 25개 ID만 필터링
with open("results/main_exp/final_vf_scores_v2.json") as f:
    all_v2 = json.load(f)

ai_data = {
    f"{r['id']}_{r['cf_type']}": r
    for r in all_v2
    if r["id"] in sample_ids  # 반드시 25개 ID만
}
print(f"AI 점수 (v2, 25개 필터링): {len(ai_data)}개")

CF_TYPES = ["semantic_swap", "attribute_flip", "random", "masked"]

# ── 정규화 함수 ───────────────────────────────────────────
def normalize(score, min_val=1, max_val=5):
    return (score - min_val) / (max_val - min_val)

# ── VF-Score 계산 ─────────────────────────────────────────
ALPHA = 0.05

def compute_vf_score(sim_score, faith_score, alpha=ALPHA):
    sim_norm   = normalize(sim_score)
    faith_norm = normalize(faith_score)
    return alpha * (1 - sim_norm) + (1 - alpha) * faith_norm

# ── 인간 점수 파싱 + sim 반전 ─────────────────────────────
# HTML 툴: sim 5=똑같음(나쁨) → 반전: 6-sim
human_data = {}
for item_id in sample_ids:
    for cf_type in CF_TYPES:
        sim_key   = f"sim_{item_id}_{cf_type}"
        faith_key = f"faith_{item_id}_{cf_type}"
        if sim_key in human_raw and faith_key in human_raw:
            raw_sim   = human_raw[sim_key]
            raw_faith = human_raw[faith_key]
            flipped_sim = 6 - raw_sim  # 방향 통일
            human_data[f"{item_id}_{cf_type}"] = {
                "sim_score":   flipped_sim,
                "faith_score": raw_faith,
            }

print(f"인간 점수 파싱: {len(human_data)}개")

# ── 공통 키 확인 ──────────────────────────────────────────
common_keys = list(set(human_data.keys()) & set(ai_data.keys()))
print(f"공통 키: {len(common_keys)}개 / 100개")

# ── VF-Score 계산 & 비교 ──────────────────────────────────
human_vf_scores = []
ai_vf_scores    = []

for key in common_keys:
    h = human_data[key]
    a = ai_data[key]

    h_vf = compute_vf_score(h["sim_score"], h["faith_score"])
    a_vf = a["vf_score"]  # 이미 v2에서 계산된 값

    human_vf_scores.append(h_vf)
    ai_vf_scores.append(a_vf)

# ── 상관관계 계산 ─────────────────────────────────────────
spearman_r, spearman_p = spearmanr(human_vf_scores, ai_vf_scores)
pearson_r,  pearson_p  = pearsonr(human_vf_scores,  ai_vf_scores)

print(f"\n=== VF-Score v2 vs Human Annotation 상관관계 ===")
print(f"샘플 수:    {len(common_keys)}개")
print(f"Spearman ρ: {spearman_r:.4f} (p={spearman_p:.4f})")
print(f"Pearson  r: {pearson_r:.4f}  (p={pearson_p:.4f})")

if spearman_r > 0.5:
    print("\n✅ 강한 상관관계 — VF-Score v2 유효성 확인")
elif spearman_r > 0.3:
    print("\n⚠️ 보통 상관관계 — v1보다 개선됨")
elif spearman_r > 0:
    print("\n⚠️ 약한 상관관계 — 추가 개선 필요")
else:
    print("\n❌ 음의 상관관계")

# ── v1과 비교 ─────────────────────────────────────────────
print("\n=== v1 vs v2 비교 ===")
print(f"v1 Spearman ρ: 0.3007 (이미지 없이 faith 계산)")
print(f"v2 Spearman ρ: {spearman_r:.4f} (이미지 포함 faith 계산)")
improvement = spearman_r - 0.3007
print(f"개선폭:        {improvement:+.4f}")

# ── CF 유형별 상관관계 ────────────────────────────────────
print("\n=== CF 유형별 상관관계 ===")
print(f"{'CF Type':<16} {'N':>4} {'Spearman ρ':>12} {'p-value':>10}")
print("-" * 46)
for cf_type in CF_TYPES:
    cf_keys = [k for k in common_keys if k.endswith(f"_{cf_type}")]
    if len(cf_keys) < 3:
        continue
    h_vf = [compute_vf_score(human_data[k]["sim_score"],
                              human_data[k]["faith_score"]) for k in cf_keys]
    a_vf = [ai_data[k]["vf_score"] for k in cf_keys]
    r, p = spearmanr(h_vf, a_vf)
    print(f"{cf_type:<16} {len(cf_keys):>4} {r:>12.4f} {p:>10.4f}")

# ── 결과 저장 ─────────────────────────────────────────────
result = {
    "n_samples":   len(common_keys),
    "alpha":       ALPHA,
    "spearman_r":  round(spearman_r, 4),
    "spearman_p":  round(spearman_p, 4),
    "pearson_r":   round(pearson_r,  4),
    "pearson_p":   round(pearson_p,  4),
    "v1_spearman": 0.3007,
    "improvement": round(improvement, 4),
}
with open("results/main_exp/vf_score_v2_validation.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\n저장 완료 → results/main_exp/vf_score_v2_validation.json")