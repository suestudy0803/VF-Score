# ~/vf-cot/scripts/method2_generate_cot.py
# 실행: cd ~/vf-cot/scripts && python method2_generate_cot.py

import torch, json, os
from PIL import Image
from datasets import load_dataset
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from tqdm import tqdm
from pathlib import Path

MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"
N_SAMPLES  = 300  # medium에서 300개 사용 (비용/시간 절약)

PROMPT_IMG1 = """[INST] <image>
Examine this image carefully and describe what you observe step by step.
Then answer: what are the key visual elements in this image?
Let's think step by step: [/INST]"""

PROMPT_IMG2 = """[INST] <image>
Examine this image carefully and describe what you observe step by step.
Then answer: what are the key visual elements in this image?
Let's think step by step: [/INST]"""

# ── 데이터 로드 ───────────────────────────────────────────
print("SPD-Faith Bench medium split 로드 중...")
ds = load_dataset("Jackson-Lv/SPD-Faith-Bench", split="medium")
ds = ds.select(range(N_SAMPLES))
print(f"샘플: {N_SAMPLES}개")

Path("data/spd_finetune").mkdir(parents=True, exist_ok=True)

# ── 모델 로드 ─────────────────────────────────────────────
print("모델 로드 중...")
processor = LlavaNextProcessor.from_pretrained(MODEL_PATH)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# ── CoT 생성 함수 ─────────────────────────────────────────
def generate_cot(image, prompt):
    inputs = processor(
        text=prompt, images=image, return_tensors="pt"
    ).to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )
    result = processor.decode(output[0], skip_special_tokens=True)
    return result.split("Let's think step by step:")[-1].strip()

# ── 이어서 실행 ───────────────────────────────────────────
out_path = "data/spd_finetune/cot_pairs.json"
done = set()
results = []
if os.path.exists(out_path):
    with open(out_path) as f:
        results = json.load(f)
        done = {r["image_id"] for r in results}
    print(f"이어서 실행: {len(done)}개 완료")

# ── 메인 루프 ─────────────────────────────────────────────
print("CoT 생성 중 (image1 + image2 각각)...")
for item in tqdm(ds):
    image_id = str(item["image_id"])
    if image_id in done:
        continue

    try:
        img1 = item["image1"].convert("RGB")
        img2 = item["image2"].convert("RGB")

        cot1 = generate_cot(img1, PROMPT_IMG1)
        cot2 = generate_cot(img2, PROMPT_IMG2)

        results.append({
            "image_id":      image_id,
            "cot_img1":      cot1,   # 원본 이미지 CoT
            "cot_img2":      cot2,   # 변형 이미지 CoT
            "ground_truth":  item["differences"],
            "n_differences": item["num_differences"],
        })
    except Exception as e:
        print(f"  에러 [{image_id}]: {e}")

    if len(results) % 50 == 0:
        with open(out_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  {len(results)}개 저장")

with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n완료: {len(results)}개 → {out_path}")