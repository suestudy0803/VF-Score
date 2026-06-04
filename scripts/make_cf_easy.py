# ~/vf-cot/scripts/make_cf_easy.py
import json, os, random
from PIL import Image
from datasets import load_dataset

with open("data/raw/subset_500.json") as f:
    data = json.load(f)

# COCO 이미지 랜덤 풀 준비
coco = load_dataset("detection-datasets/coco", split="train", streaming=True)
coco_images = []
for i, item in enumerate(coco):
    coco_images.append(item["image"])
    if i >= 1000:
        break
print("COCO 이미지 준비 완료")

os.makedirs("data/counterfactual/random", exist_ok=True)
os.makedirs("data/counterfactual/masked", exist_ok=True)

results = []
for item in data:
    item_id = item["id"]
    orig = Image.open(item["image_path"]).convert("RGB")

    # ③ Random — COCO에서 랜덤 이미지
    rand_img = random.choice(coco_images).convert("RGB")
    rand_img = rand_img.resize(orig.size)
    rand_path = f"data/counterfactual/random/{item_id}.jpg"
    rand_img.save(rand_path)

    # ④ Masked — 검은 화면
    masked = Image.new("RGB", orig.size, (0, 0, 0))
    mask_path = f"data/counterfactual/masked/{item_id}.jpg"
    masked.save(mask_path)

    results.append({
        "id": item_id,
        "original": item["image_path"],
        "random": rand_path,
        "masked": mask_path,
    })

with open("data/counterfactual/cf_paths.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"완료: {len(results)}개 → data/counterfactual/")