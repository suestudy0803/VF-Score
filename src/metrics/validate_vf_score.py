# validate_vf_score.py 수정 — 모든 CF type VF-Score 평균 사용
import json
import numpy as np
from scipy.stats import spearmanr, pearsonr

with open("/home/hufs/vf-cot/scripts/data/annotations/annotation_result.json") as f:
    annotations = json.load(f)

with open("/home/hufs/vf-cot/scripts/data/annotations/samples_50.json") as f:
    samples = json.load(f)

# 4종류 VF-Score 평균으로 계산 (더 robust)
vf_all = {}
for cf_type in ["semantic_swap", "attribute_flip", "random", "masked"]:
    path = f"results/main_exp/vf_scores_{cf_type}.json"
    with open(path) as f:
        for item in json.load(f):
            if item["id"] not in vf_all:
                vf_all[item["id"]] = []
            vf_all[item["id"]].append(item["vf_score"])

# 평균 VF-Score
vf_mean = {id: np.mean(scores) for id, scores in vf_all.items()}

sample_human_scores = []
sample_vf_scores    = []

for i, sample in enumerate(samples):
    item_id = sample["id"]

    step_labels = []
    for j in range(4):
        key = f"{i}_{j}"
        if key in annotations:
            step_labels.append(annotations[key])

    if not step_labels:
        continue

    # ID가 vf에 없으면 건너뜀
    if item_id not in vf_mean:
        continue

    human_faithful = np.mean([(5 - l) / 4 for l in step_labels])
    sample_human_scores.append(human_faithful)
    sample_vf_scores.append(vf_mean[item_id])

spearman_r, spearman_p = spearmanr(sample_human_scores, sample_vf_scores)
pearson_r,  pearson_p  = pearsonr(sample_human_scores, sample_vf_scores)

print(f"매핑된 샘플: {len(sample_human_scores)}개 / 50개")
print(f"Spearman ρ: {spearman_r:.4f} (p={spearman_p:.4f})")
print(f"Pearson  r: {pearson_r:.4f}  (p={pearson_p:.4f})")

result = {
    "n_samples":  len(sample_human_scores),
    "spearman_r": round(spearman_r, 4),
    "spearman_p": round(spearman_p, 4),
    "pearson_r":  round(pearson_r, 4),
    "pearson_p":  round(pearson_p, 4),
}
with open("results/main_exp/vf_score_validation.json", "w") as f:
    json.dump(result, f, indent=2)
print("저장 완료")

