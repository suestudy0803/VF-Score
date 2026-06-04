# ~/vf-cot/scripts/run_cf_cot_inference.py
import torch, json, os
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from tqdm import tqdm

MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"
COT_PROMPT = """[INST] <image>
Examine the image carefully and answer the following question step by step.

Question: {question}

Instructions:
1. First, describe what you observe in the image relevant to the question.
2. Then, reason step by step based on what you see.
3. Finally, provide your answer.

Let's think step by step: [/INST]"""

# 모델 로드
print("모델 로딩 중...")
processor = LlavaNextProcessor.from_pretrained(MODEL_PATH)
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
)
model.eval()
print("로드 완료")

# 데이터 로드
with open("data/raw/subset_500.json") as f:
    raw = {item["id"]: item for item in json.load(f)}

with open("data/counterfactual/cf_openai.json") as f:
    cf_openai = {item["id"]: item for item in json.load(f)}

with open("data/counterfactual/cf_paths.json") as f:
    cf_easy = {item["id"]: item for item in json.load(f)}

# 4종류 이미지 타입 정의
CF_TYPES = {
    "semantic_swap":    lambda id: cf_openai.get(id, {}).get("semantic_swap"),
    "attribute_flip":   lambda id: cf_openai.get(id, {}).get("attribute_flip"),
    "random":           lambda id: cf_easy.get(id, {}).get("random"),
    "masked":           lambda id: cf_easy.get(id, {}).get("masked"),
}

def generate_cot(image, question):
    prompt = COT_PROMPT.format(question=question)
    inputs = processor(
        text=prompt, images=image, return_tensors="pt"
    ).to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
        )
    result = processor.decode(output[0], skip_special_tokens=True)
    return result.split("Let's think step by step:")[-1].strip()

# cf_openai에 있는 ID만 사용 (200개)
target_ids = list(cf_openai.keys())
print(f"대상: {len(target_ids)}개")

os.makedirs("data/cot_outputs", exist_ok=True)

for cf_type, path_fn in CF_TYPES.items():
    out_path = f"data/cot_outputs/cf_cot_{cf_type}.json"

    # 이어서 실행
    done = set()
    results = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
            done = {r["id"] for r in results}
        print(f"[{cf_type}] 이어서: {len(done)}개 완료")

    print(f"\n[{cf_type}] 시작...")
    for item_id in tqdm(target_ids):
        if item_id in done:
            continue

        img_path = path_fn(item_id)
        if not img_path or not os.path.exists(img_path):
            continue

        question = raw[item_id]["question"]
        image = Image.open(img_path).convert("RGB")
        cot = generate_cot(image, question)

        results.append({
            "id": item_id,
            "cf_type": cf_type,
            "question": question,
            "image_path": img_path,
            "cot": cot,
        })

        if len(results) % 50 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[{cf_type}] 완료: {len(results)}개 → {out_path}")

print("\n전체 완료!")