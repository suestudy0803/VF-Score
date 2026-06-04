# ~/vf-cot/src/metrics/vf_score.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr
from tqdm import tqdm
import os

# Sentence-BERT 로드
model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_vf_score(cot_orig: str, cot_cf: str) -> float:
    """
    VF-Score = 1 - cosine_similarity(CoT_원본, CoT_변경)
    높을수록 이미지 변화에 민감하게 반응 = faithful
    """
    emb_orig = model.encode([cot_orig])
    emb_cf   = model.encode([cot_cf])
    sim = float(np.dot(emb_orig[0], emb_cf[0]) /
                (np.linalg.norm(emb_orig[0]) * np.linalg.norm(emb_cf[0])))
    return round(1 - sim, 4)

def compute_all_vf_scores(orig_path, cf_type):
    """원본 CoT와 CF CoT를 매핑해서 VF-Score 계산"""
    cf_path = f"scripts/data/cot_outputs/cf_cot_{cf_type}.json"

    with open(orig_path) as f:
        orig_data = {item["id"]: item for item in json.load(f)}
    with open(cf_path) as f:
        cf_data = {item["id"]: item for item in json.load(f)}

    common_ids = list(set(orig_data.keys()) & set(cf_data.keys()))
    print(f"[{cf_type}] 공통 ID: {len(common_ids)}개")

    results = []
    for item_id in tqdm(common_ids):
        cot_orig = orig_data[item_id]["cot"]
        cot_cf   = cf_data[item_id]["cot"]
        score    = compute_vf_score(cot_orig, cot_cf)

        results.append({
            "id":       item_id,
            "cf_type":  cf_type,
            "vf_score": score,
            "cot_orig": cot_orig[:200],  # 저장 공간 절약
            "cot_cf":   cot_cf[:200],
        })

    return results

# ── 메인 실행 ──────────────────────────────────────────
os.makedirs("results/main_exp", exist_ok=True)

CF_TYPES = ["semantic_swap", "attribute_flip", "random", "masked"]
ORIG_PATH = "scripts/data/cot_outputs/llava_cot_500.json"

all_results = {}
summary = {}

for cf_type in CF_TYPES:
    print(f"\n=== {cf_type} ===")
    results = compute_all_vf_scores(ORIG_PATH, cf_type)
    all_results[cf_type] = results

    scores = [r["vf_score"] for r in results]
    summary[cf_type] = {
        "n":    len(scores),
        "mean": round(float(np.mean(scores)), 4),
        "std":  round(float(np.std(scores)), 4),
        "min":  round(float(np.min(scores)), 4),
        "max":  round(float(np.max(scores)), 4),
    }

    # 저장
    out_path = f"results/main_exp/vf_scores_{cf_type}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"저장 → {out_path}")
    print(f"평균 VF-Score: {summary[cf_type]['mean']} ± {summary[cf_type]['std']}")

# 요약 저장
with open("results/main_exp/vf_score_summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n\n=== 전체 요약 ===")
print(f"{'CF Type':<20} {'N':>6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 60)
for cf_type, s in summary.items():
    print(f"{cf_type:<20} {s['n']:>6} {s['mean']:>8} {s['std']:>8} {s['min']:>8} {s['max']:>8}")


