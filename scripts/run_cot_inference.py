# ~/vf-cot/scripts/run_cot_inference.py
import torch
import json
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from tqdm import tqdm
import os

MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"
DATA_PATH  = "data/raw/subset_500.json"
OUT_PATH   = "data/cot_outputs/llava_cot_500.json"

COT_PROMPT = """[INST] <image>
Examine the image carefully and answer the following question step by step.

Question: {question}

Instructions:
1. First, describe what you observe in the image relevant to the question.
2. Then, reason step by step based on what you see.
3. Finally, state your answer clearly.

Let's think step by step: [/INST]"""

# 모델 로드
print("모델 로딩 중...")
processor = LlavaNextProcessor.from_pretrained(MODEL_PATH)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,       # RTX 5090 권장
    device_map="auto",
    load_in_4bit=True,
    attn_implementation="flash_attention_2",
)
model.eval()
print("로드 완료")

# 이어서 실행 가능하게 (중간에 꺼져도 OK)
done = set()
results = []
os.makedirs("data/cot_outputs", exist_ok=True)
if os.path.exists(OUT_PATH):
    with open(OUT_PATH) as f:
        results = json.load(f)
        done = {r["id"] for r in results}
    print(f"이어서 실행: {len(done)}개 이미 완료")

# 데이터 로드
with open(DATA_PATH) as f:
    data = json.load(f)

# Inference 루프
for item in tqdm(data):
    if item["id"] in done:
        continue

    image = Image.open(item["image_path"]).convert("RGB")
    prompt = COT_PROMPT.format(question=item["question"])

    inputs = processor(
        text=prompt, images=image, return_tensors="pt"
    ).to("cuda")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
        )

    cot = processor.decode(output[0], skip_special_tokens=True)
    # 프롬프트 부분 제거하고 응답만 저장
    cot_only = cot.split("Let's think step by step:")[-1].strip()

    results.append({
        "id": item["id"],
        "question": item["question"],
        "answer": item["answer"],
        "subject": item["subject"],
        "image_path": item["image_path"],
        "cot": cot_only,
        "model": "llava-1.6-7b",
    })

    # 50개마다 중간 저장
    if len(results) % 50 == 0:
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"{len(results)}개 저장 완료")

# 최종 저장
with open(OUT_PATH, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n완료: 총 {len(results)}개 → {OUT_PATH}")