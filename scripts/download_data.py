# ~/vf-cot/scripts/download_data.py
from datasets import load_dataset
import json, os

# ScienceQA 다운로드
dataset = load_dataset("derek-thomas/ScienceQA", split="test")

# 이미지가 있는 것만 필터링 (이미지 없는 문제 제외)
has_image = [i for i, x in enumerate(dataset) if x["image"] is not None]
print(f"전체: {len(dataset)}개 / 이미지 있는 것: {len(has_image)}개")

# 500개 서브셋 선정 (subject 균형 맞춰서)
from collections import defaultdict
import random
random.seed(42)

by_subject = defaultdict(list)
for i in has_image:
    by_subject[dataset[i]["subject"]].append(i)

# subject별 균등 샘플링
subset_ids = []
for subj, ids in by_subject.items():
    n = min(170, len(ids))
    subset_ids.extend(random.sample(ids, n))

subset_ids = subset_ids[:500]
print(f"서브셋: {len(subset_ids)}개")

# 저장
os.makedirs("data/raw", exist_ok=True)
subset = []
for i in subset_ids:
    item = dataset[i]
    img_path = f"data/raw/images/{i}.jpg"
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    item["image"].save(img_path)
    subset.append({
        "id": str(i),
        "question": item["question"],
        "answer": item["answer"],
        "subject": item["subject"],
        "image_path": img_path,
    })

with open("data/raw/subset_500.json", "w") as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)

print("저장 완료 → data/raw/subset_500.json")