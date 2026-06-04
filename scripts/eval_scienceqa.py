# ~/vf-cot/scripts/eval_scienceqa.py
# 실행: cd ~/vf-cot/scripts && python eval_scienceqa.py

import torch, json, os, time
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
import numpy as np

MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"
LORA_PATH  = "models/checkpoints/llava-vf-lora/final"
client     = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

COT_PROMPT = """[INST] <image>
Examine the image carefully and answer the following question step by step.

Instructions:
1. First, describe what you observe in the image relevant to the question.
2. Then, reason step by step based on what you see.
3. Finally, provide your answer.

Question: {question}

Let's think step by step: [/INST]"""

# ── 데이터 로드 ───────────────────────────────────────────
# 파인튜닝에 쓰지 않은 나머지 데이터로 평가
with open("data/raw/subset_500.json") as f:
    all_data = json.load(f)

with open("results/main_exp/filtered_ids_high.json") as f:
    train_ids = set(json.load(f))

# 파인튜닝에 안 쓴 78개 (Low VF-Score) + 나머지로 테스트셋 구성
# 완전히 새로운 테스트셋: 파인튜닝에 쓰인 122개 제외한 나머지
test_data = [item for item in all_data if item["id"] not in train_ids]
print(f"테스트셋: {len(test_data)}개 (파인튜닝 미사용 데이터)")

Path("results/scienceqa_eval").mkdir(parents=True, exist_ok=True)

# ── 응답 생성 함수 ────────────────────────────────────────
def generate_cot(model, processor, image, question):
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

# ── 정답 추출 함수 (GPT-4o) ───────────────────────────────
def extract_answer(question, cot, choices):
    choices_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
    prompt = f"""다음 질문과 추론을 보고 최종 답변의 번호(1~{len(choices)})만 출력하세요.

질문: {question}
선택지:
{choices_str}

추론: {cot[:500]}

번호만 출력하세요. 다른 텍스트 없이."""
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        import re
        match = re.search(r'[1-9]', res.choices[0].message.content)
        return int(match.group()) - 1 if match else -1
    except:
        return -1

# ── 프로세서 로드 ─────────────────────────────────────────
print("프로세서 로드 중...")
processor = LlavaNextProcessor.from_pretrained(MODEL_PATH)

# ── Baseline 평가 ─────────────────────────────────────────
print("\nBaseline 모델 로드 중...")
baseline_model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)
baseline_model.eval()

baseline_out = "results/scienceqa_eval/baseline_results.json"
done = set()
baseline_results = []

if os.path.exists(baseline_out):
    with open(baseline_out) as f:
        baseline_results = json.load(f)
        done = {r["id"] for r in baseline_results}

print(f"[1/2] Baseline 평가 중... ({len(test_data)}개)")
for item in tqdm(test_data):
    if item["id"] in done:
        continue
    try:
        image    = Image.open(item["image_path"]).convert("RGB")
        question = item["question"]
        cot      = generate_cot(baseline_model, processor, image, question)
        pred_idx = extract_answer(question, cot, item.get("choices", []))
        correct  = (pred_idx == item.get("answer", -1))

        baseline_results.append({
            "id":       item["id"],
            "question": question,
            "cot":      cot,
            "pred":     pred_idx,
            "answer":   item.get("answer", -1),
            "correct":  correct,
            "model":    "baseline",
        })
        time.sleep(0.5)
    except Exception as e:
        print(f"  에러 [{item['id']}]: {e}")

    if len(baseline_results) % 20 == 0:
        with open(baseline_out, "w") as f:
            json.dump(baseline_results, f, ensure_ascii=False, indent=2)

with open(baseline_out, "w") as f:
    json.dump(baseline_results, f, ensure_ascii=False, indent=2)
print(f"Baseline 완료: {len(baseline_results)}개")

# ── Fine-tuned 평가 ───────────────────────────────────────
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

ft_out = "results/scienceqa_eval/finetuned_results.json"
done = set()
ft_results = []

if os.path.exists(ft_out):
    with open(ft_out) as f:
        ft_results = json.load(f)
        done = {r["id"] for r in ft_results}

print(f"[2/2] Fine-tuned 평가 중... ({len(test_data)}개)")
for item in tqdm(test_data):
    if item["id"] in done:
        continue
    try:
        image    = Image.open(item["image_path"]).convert("RGB")
        question = item["question"]
        cot      = generate_cot(ft_model, processor, image, question)
        pred_idx = extract_answer(question, cot, item.get("choices", []))
        correct  = (pred_idx == item.get("answer", -1))

        ft_results.append({
            "id":       item["id"],
            "question": question,
            "cot":      cot,
            "pred":     pred_idx,
            "answer":   item.get("answer", -1),
            "correct":  correct,
            "model":    "finetuned",
        })
        time.sleep(0.5)
    except Exception as e:
        print(f"  에러 [{item['id']}]: {e}")

    if len(ft_results) % 20 == 0:
        with open(ft_out, "w") as f:
            json.dump(ft_results, f, ensure_ascii=False, indent=2)

with open(ft_out, "w") as f:
    json.dump(ft_results, f, ensure_ascii=False, indent=2)
print(f"Fine-tuned 완료: {len(ft_results)}개")

# ── 최종 비교 ─────────────────────────────────────────────
from scipy.stats import ttest_rel

with open(baseline_out) as f:
    b = json.load(f)
with open(ft_out) as f:
    ft = json.load(f)

b_dict  = {r["id"]: r for r in b}
ft_dict = {r["id"]: r for r in ft}
common  = list(set(b_dict.keys()) & set(ft_dict.keys()))

b_acc  = np.mean([b_dict[id]["correct"] for id in common])
ft_acc = np.mean([ft_dict[id]["correct"] for id in common])

b_correct  = [int(b_dict[id]["correct"])  for id in common]
ft_correct = [int(ft_dict[id]["correct"]) for id in common]
_, p_val   = ttest_rel(ft_correct, b_correct)

print(f"\n=== ScienceQA 평가 결과 ===")
print(f"샘플 수:           {len(common)}개")
print(f"Baseline 정확도:   {b_acc:.4f} ({b_acc*100:.1f}%)")
print(f"Fine-tuned 정확도: {ft_acc:.4f} ({ft_acc*100:.1f}%)")
print(f"개선폭:            {ft_acc - b_acc:+.4f}")
print(f"t-test p-value:    {p_val:.4f}")

summary = {
    "n_samples":      len(common),
    "baseline_acc":   round(b_acc, 4),
    "finetuned_acc":  round(ft_acc, 4),
    "improvement":    round(ft_acc - b_acc, 4),
    "ttest_p":        round(p_val, 4),
}
with open("results/scienceqa_eval/comparison.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"저장 완료 → results/scienceqa_eval/comparison.json")