# ~/vf-cot/scripts/generate_spd_responses.py
# 실행: cd ~/vf-cot/scripts && python generate_spd_responses.py

import torch, json, os
from PIL import Image
from datasets import load_dataset
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm
from pathlib import Path

MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"
LORA_PATH  = "models/checkpoints/llava-vf-lora/final"
SPLIT      = "easy"
N_SAMPLES  = 100

PROMPT = """[INST] <image>
You are given two images side by side. The left image is the original and the right image is the modified version.
Look carefully at both images and describe the differences step by step.

Instructions:
1. First, describe what you observe in the original image (left).
2. Then, identify what has changed in the modified image (right).
3. Finally, list all differences you found clearly.

Let's think step by step: [/INST]"""

# ── 두 이미지를 나란히 합치는 함수 ───────────────────────
def merge_images(img1, img2):
    """image1(원본)과 image2(변형)를 좌우로 합쳐 하나의 이미지로"""
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")

    # 같은 높이로 맞춤
    h = max(img1.height, img2.height)
    img1 = img1.resize((int(img1.width * h / img1.height), h))
    img2 = img2.resize((int(img2.width * h / img2.height), h))

    # 좌우 합치기
    merged = Image.new("RGB", (img1.width + img2.width, h))
    merged.paste(img1, (0, 0))
    merged.paste(img2, (img1.width, 0))
    return merged

# ── 응답 생성 함수 ────────────────────────────────────────
def generate_response(model, processor, image, max_new_tokens=400):
    inputs = processor(
        text=PROMPT,
        images=image,
        return_tensors="pt"
    ).to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    result = processor.decode(output[0], skip_special_tokens=True)
    return result.split("Let's think step by step:")[-1].strip()

# ── 데이터 로드 ───────────────────────────────────────────
print("SPD-Faith Bench 로드 중...")
ds = load_dataset("Jackson-Lv/SPD-Faith-Bench", split=SPLIT)
ds = ds.select(range(N_SAMPLES))
print(f"샘플: {N_SAMPLES}개")

Path("results/spd_bench").mkdir(parents=True, exist_ok=True)

# ── 프로세서 로드 ─────────────────────────────────────────
print("프로세서 로드 중...")
processor = LlavaNextProcessor.from_pretrained(MODEL_PATH)

# ── Baseline 모델 로드 & 응답 생성 ───────────────────────
print("\nBaseline 모델 로드 중...")
baseline_model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)
baseline_model.eval()

baseline_out     = "results/spd_bench/baseline_responses.json"
done             = set()
baseline_results = []

if os.path.exists(baseline_out):
    with open(baseline_out) as f:
        baseline_results = json.load(f)
        done = {r["image_id"] for r in baseline_results}

print("[1/2] Baseline 응답 생성 중...")
for item in tqdm(ds):
    image_id = str(item["image_id"])
    if image_id in done:
        continue

    # 두 이미지 합치기
    merged = merge_images(item["image1"], item["image2"])
    response = generate_response(baseline_model, processor, merged)

    # ground truth 파싱
    gt = [
        {
            "type":        d.get("type", ""),
            "category":    d.get("category", ""),
            "description": d.get("description", ""),
        }
        for d in item["differences"]
    ]

    baseline_results.append({
        "image_id":      image_id,
        "response":      response,
        "ground_truth":  gt,
        "n_differences": item["num_differences"],
        "model":         "baseline",
    })

    if len(baseline_results) % 20 == 0:
        with open(baseline_out, "w") as f:
            json.dump(baseline_results, f, ensure_ascii=False, indent=2)
        print(f"  {len(baseline_results)}개 저장")

with open(baseline_out, "w") as f:
    json.dump(baseline_results, f, ensure_ascii=False, indent=2)
print(f"Baseline 완료: {len(baseline_results)}개 → {baseline_out}")

# ── Fine-tuned 모델 로드 & 응답 생성 ─────────────────────
print("\nFine-tuned 모델 로드 중...")
del baseline_model
torch.cuda.empty_cache()

base_model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)
ft_model = PeftModel.from_pretrained(base_model, LORA_PATH)
ft_model.eval()

ft_out     = "results/spd_bench/finetuned_responses.json"
done       = set()
ft_results = []

if os.path.exists(ft_out):
    with open(ft_out) as f:
        ft_results = json.load(f)
        done = {r["image_id"] for r in ft_results}

print("[2/2] Fine-tuned 응답 생성 중...")
for item in tqdm(ds):
    image_id = str(item["image_id"])
    if image_id in done:
        continue

    merged   = merge_images(item["image1"], item["image2"])
    response = generate_response(ft_model, processor, merged)

    gt = [
        {
            "type":        d.get("type", ""),
            "category":    d.get("category", ""),
            "description": d.get("description", ""),
        }
        for d in item["differences"]
    ]

    ft_results.append({
        "image_id":      image_id,
        "response":      response,
        "ground_truth":  gt,
        "n_differences": item["num_differences"],
        "model":         "finetuned",
    })

    if len(ft_results) % 20 == 0:
        with open(ft_out, "w") as f:
            json.dump(ft_results, f, ensure_ascii=False, indent=2)
        print(f"  {len(ft_results)}개 저장")

with open(ft_out, "w") as f:
    json.dump(ft_results, f, ensure_ascii=False, indent=2)
print(f"Fine-tuned 완료: {len(ft_results)}개 → {ft_out}")

print("\n=== 완료 ===")
print(f"다음: eval_cot_faithfulness.py로 DRF 점수 측정")