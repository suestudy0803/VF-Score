# ~/vf-cot/scripts/prepare_finetune_data.py
# 실행: cd ~/vf-cot/scripts && python prepare_finetune_data.py

import json
import os
from pathlib import Path

# ── 데이터 로드 ───────────────────────────────────────────
with open("results/main_exp/filtered_ids_high.json") as f:
    high_ids = set(json.load(f))

with open("data/cot_outputs/llava_cot_500.json") as f:
    orig_cot = {item["id"]: item for item in json.load(f)}

print(f"High VF-Score ID: {len(high_ids)}개")

# ── LLaVA SFT 형식으로 변환 ───────────────────────────────
# LLaVA fine-tuning 형식:
# {"id": "...", "image": "path/to/image.jpg",
#  "conversations": [
#    {"from": "human", "value": "<image>\n{question}"},
#    {"from": "gpt",   "value": "{cot}"}
#  ]}

COT_INSTRUCTION = (
    "Examine the image carefully and answer the following "
    "question step by step.\n\n"
    "Instructions:\n"
    "1. First, describe what you observe in the image relevant to the question.\n"
    "2. Then, reason step by step based on what you see.\n"
    "3. Finally, provide your answer.\n\n"
    "Let's think step by step:"
)

train_data = []
skipped    = 0

for item_id in high_ids:
    item = orig_cot.get(item_id)
    if not item:
        skipped += 1
        continue

    img_path = item.get("image_path", "")
    question  = item.get("question", "")
    cot       = item.get("cot", "")

    if not img_path or not question or not cot:
        skipped += 1
        continue

    # 절대경로로 변환
    abs_img_path = os.path.abspath(img_path)
    if not os.path.exists(abs_img_path):
        skipped += 1
        continue

    train_data.append({
        "id": item_id,
        "image": abs_img_path,
        "conversations": [
            {
                "from":  "human",
                "value": f"<image>\n{COT_INSTRUCTION}\n\nQuestion: {question}"
            },
            {
                "from":  "gpt",
                "value": cot
            }
        ]
    })

print(f"학습 데이터: {len(train_data)}개 / 스킵: {skipped}개")

# ── 저장 ──────────────────────────────────────────────────
Path("data/finetune").mkdir(exist_ok=True)
out_path = "data/finetune/train_high_vf.json"
with open(out_path, "w") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

print(f"저장 완료 → {out_path}")
print(f"\n샘플 확인:")
print(f"  ID: {train_data[0]['id']}")
print(f"  Image: {train_data[0]['image']}")
print(f"  Q: {train_data[0]['conversations'][0]['value'][:100]}...")
print(f"  A: {train_data[0]['conversations'][1]['value'][:100]}...")