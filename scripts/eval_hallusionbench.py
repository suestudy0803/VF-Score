# ~/vf-cot/scripts/eval_hallusionbench.py
# 실행: cd ~/vf-cot/scripts && python eval_hallusionbench.py

import torch, json, os
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm
from pathlib import Path

MODEL_PATH  = "llava-hf/llava-v1.6-mistral-7b-hf"
LORA_PATH   = "models/checkpoints/llava-vf-lora/final"
BENCH_DIR   = "../HallusionBench"
BENCH_JSON  = "../HallusionBench/HallusionBench.json"

PROMPT = """[INST] <image>
Please look at the image carefully and answer the following question.
Answer only with "Yes" or "No".

Question: {question} [/INST]"""

PROMPT_NO_IMG = """[INST]
Please answer the following question.
Answer only with "Yes" or "No".

Question: {question} [/INST]"""

# ── 데이터 로드 ───────────────────────────────────────────
with open(BENCH_JSON) as f:
    data = json.load(f)

# 이미지 있는 것만
data = [d for d in data if d.get("visual_input") == "1"
        and d.get("filename") is not None]
print(f"평가 대상: {len(data)}개 (이미지 있는 것만)")

Path("results/hallusionbench").mkdir(parents=True, exist_ok=True)

# ── 응답 생성 함수 ────────────────────────────────────────
def generate_answer(model, processor, item):
    img_path = os.path.join(BENCH_DIR, item["filename"])

    try:
        image  = Image.open(img_path).convert("RGB")
        prompt = PROMPT.format(question=item["question"])
        inputs = processor(
            text=prompt, images=image,
            return_tensors="pt"
        ).to("cuda")
    except:
        # 이미지 로드 실패 시 텍스트만
        prompt = PROMPT_NO_IMG.format(question=item["question"])
        inputs = processor(
            text=prompt, return_tensors="pt"
        ).to("cuda")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )
    result = processor.decode(output[0], skip_special_tokens=True)

    # Yes/No 추출
    answer = result.split("[/INST]")[-1].strip().lower()
    if "yes" in answer:
        return "1"
    elif "no" in answer:
        return "0"
    else:
        return "2"  # Uncertain

# ── 평가 실행 함수 ────────────────────────────────────────
def run_eval(model, processor, model_name, out_path):
    done = set()
    results = []

    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
            done = {r["question_id"] for r in results}
        print(f"[{model_name}] 이어서: {len(done)}개 완료")

    print(f"[{model_name}] 평가 중... ({len(data)}개)")
    for item in tqdm(data):
        qid = str(item["question_id"])
        if qid in done:
            continue

        pred = generate_answer(model, processor, item)

        results.append({
            "question_id":      item["question_id"],
            "category":         item["category"],
            "subcategory":      item["subcategory"],
            "set_id":           item["set_id"],
            "figure_id":        item["figure_id"],
            "question":         item["question"],
            "gt_answer":        item["gt_answer"],
            "model_prediction": pred,
            "filename":         item["filename"],
            "model":            model_name,
        })

        if len(results) % 50 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[{model_name}] 완료: {len(results)}개 → {out_path}")
    return results

# ── 정확도 계산 함수 ──────────────────────────────────────
def calc_accuracy(results):
    correct = sum(1 for r in results
                  if r["model_prediction"] == r["gt_answer"])
    total   = len(results)
    return correct / total if total > 0 else 0

def calc_figure_accuracy(results):
    """Figure 단위 정확도: 같은 figure_id의 모든 질문을 맞춰야 정답"""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        key = f"{r['category']}_{r['set_id']}_{r['figure_id']}"
        groups[key].append(r["model_prediction"] == r["gt_answer"])
    correct = sum(1 for v in groups.values() if all(v))
    return correct / len(groups) if groups else 0

def calc_question_pair_accuracy(results):
    """Question pair 정확도: 같은 set_id 내 모든 질문을 맞춰야 정답"""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        key = f"{r['category']}_{r['set_id']}"
        groups[key].append(r["model_prediction"] == r["gt_answer"])
    correct = sum(1 for v in groups.values() if all(v))
    return correct / len(groups) if groups else 0

# ── 프로세서 로드 ─────────────────────────────────────────
print("프로세서 로드 중...")
processor = LlavaNextProcessor.from_pretrained(MODEL_PATH)

# ── Baseline 평가 ─────────────────────────────────────────
print("\nBaseline 모델 로드 중...")
baseline_model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
baseline_model.eval()
b_results = run_eval(
    baseline_model, processor,
    "baseline",
    "results/hallusionbench/baseline.json"
)
del baseline_model
torch.cuda.empty_cache()

# ── Fine-tuned 평가 ───────────────────────────────────────
print("\nFine-tuned 모델 로드 중...")
base_model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
ft_model = PeftModel.from_pretrained(base_model, LORA_PATH)
ft_model.eval()
ft_results = run_eval(
    ft_model, processor,
    "finetuned",
    "results/hallusionbench/finetuned.json"
)

# ── 최종 비교 ─────────────────────────────────────────────
import numpy as np
from scipy.stats import ttest_rel

print("\n=== HallusionBench 평가 결과 ===")
print(f"{'지표':<25} {'Baseline':>10} {'Fine-tuned':>12} {'개선폭':>8}")
print("-" * 58)

metrics = {
    "aAcc (per question)":    calc_accuracy,
    "fAcc (per figure)":      calc_figure_accuracy,
    "qAcc (per question pair)": calc_question_pair_accuracy,
}

summary = {"n_samples": len(b_results)}
for metric_name, func in metrics.items():
    b_score  = func(b_results)
    ft_score = func(ft_results)
    diff     = ft_score - b_score
    marker   = " ✅" if diff > 0 else " ❌"
    print(f"{metric_name:<25} {b_score*100:>9.2f}% {ft_score*100:>11.2f}% {diff*100:>+7.2f}%{marker}")
    summary[metric_name] = {
        "baseline":  round(b_score, 4),
        "finetuned": round(ft_score, 4),
        "improvement": round(diff, 4),
    }

# 카테고리별 분석
print("\n=== 카테고리별 aAcc ===")
print(f"{'Category':<10} {'Baseline':>10} {'Fine-tuned':>12} {'개선폭':>8}")
print("-" * 44)
for cat in ["VD", "VS"]:
    b_cat  = [r for r in b_results  if r["category"] == cat]
    ft_cat = [r for r in ft_results if r["category"] == cat]
    b_acc  = calc_accuracy(b_cat)
    ft_acc = calc_accuracy(ft_cat)
    diff   = ft_acc - b_acc
    marker = " ✅" if diff > 0 else " ❌"
    print(f"{cat:<10} {b_acc*100:>9.2f}% {ft_acc*100:>11.2f}% {diff*100:>+7.2f}%{marker}")

# 저장
import json
with open("results/hallusionbench/comparison.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\n저장 완료 → results/hallusionbench/comparison.json")