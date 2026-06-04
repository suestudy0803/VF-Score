# ~/vf-cot/scripts/analyze_filtering.py
# 실행: cd ~/vf-cot/scripts && python analyze_filtering.py

import json
import numpy as np
from pathlib import Path

with open("results/main_exp/final_vf_scores_v2.json") as f:
    data = json.load(f)

# 이미지당 4종 VF-Score 평균으로 대표값 계산
from collections import defaultdict
id_scores = defaultdict(list)
for r in data:
    id_scores[r["id"]].append(r["vf_score"])

# 이미지당 평균 VF-Score
id_mean_scores = {
    id: np.mean(scores)
    for id, scores in id_scores.items()
}

scores = list(id_mean_scores.values())
print(f"전체 이미지: {len(scores)}개")
print(f"평균 VF-Score: {np.mean(scores):.4f}")
print(f"중간값:        {np.median(scores):.4f}")
print()

print("임계값별 필터링 결과:")
print(f"{'임계값':>8} {'남는 데이터':>12} {'비율':>8} {'용도'}")
print("-" * 55)
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    n = sum(1 for s in scores if s >= t)
    pct = n / len(scores) * 100
    tag = "← 권장" if t == 0.5 else ""
    print(f"{t:>8.1f} {n:>12}개 {pct:>7.1f}% {tag}")

print()
print("임계값별 필터링 결과 (High vs Low 비교):")
threshold = 0.5
high = [id for id, s in id_mean_scores.items() if s >= threshold]
low  = [id for id, s in id_mean_scores.items() if s < threshold]
print(f"High VF-Score (>= {threshold}): {len(high)}개 → 파인튜닝에 사용")
print(f"Low  VF-Score (<  {threshold}): {len(low)}개  → 제거")

# 저장
Path("results/main_exp").mkdir(parents=True, exist_ok=True)
with open("results/main_exp/filtered_ids_high.json", "w") as f:
    json.dump(high, f, indent=2)
with open("results/main_exp/filtered_ids_low.json", "w") as f:
    json.dump(low, f, indent=2)
print(f"\n저장 완료 → filtered_ids_high.json / filtered_ids_low.json")