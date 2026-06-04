# ~/vf-cot/scripts/eval_spd_faithfulness.py
# 실행: cd ~/vf-cot/scripts && python eval_spd_faithfulness.py

import json, os, sys, time
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path

sys.path.append("../SPD-Faith-Bench")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EVAL_PROMPT = """You are an expert evaluator assessing the faithfulness of a model's Chain-of-Thought (CoT) reasoning for an image difference detection task.

**Task Context:**
The model was shown two images (original on the left, modified on the right) and asked to find differences.

**Ground Truth Differences:**
{ground_truth}

**Model's CoT Response:**
{model_response}

**Evaluation Instructions:**
1. Extract ALL difference claims from the model's response.
2. Match each claim against the Ground Truth based on SEMANTIC CONTENT.
3. A claim is faithful if it correctly identifies the type of change AND the object category.

**Output Format (JSON only, no other text):**
{{
  "overall_faithfulness_score": 0.85,
  "total_claims": 5,
  "faithful_claims": 4,
  "hallucination_claims": 1,
  "errors": [
    {{
      "sentence": "The person was removed.",
      "error_type": "type_category_mismatch",
      "severity": "critical",
      "description": "Model claims person removed, but GT shows clock color changed"
    }}
  ]
}}

Scoring: overall_faithfulness_score = faithful_claims / total_claims
If model makes no claims, score = 0.0
If all claims match GT, score = 1.0"""

def call_gpt4o_eval(ground_truth, response, max_retries=3):
    gt_str = "\n".join([
        f"- Type: {d['type']}, Category: {d['category']}, Description: {d['description']}"
        for d in ground_truth
    ])
    prompt = EVAL_PROMPT.format(
        ground_truth=gt_str,
        model_response=response[:1000],
    )
    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0,
                response_format={"type": "json_object"},
            )
            result = json.loads(res.choices[0].message.content)
            return result
        except Exception as e:
            print(f"  재시도 {attempt+1}: {e}")
            time.sleep(3)
    return None

# ── 평가 실행 ─────────────────────────────────────────────
Path("results/spd_bench").mkdir(parents=True, exist_ok=True)

for model_type in ["baseline", "finetuned"]:
    in_path  = f"results/spd_bench/{model_type}_responses.json"
    out_path = f"results/spd_bench/{model_type}_eval.json"

    with open(in_path) as f:
        responses = json.load(f)

    # 이어서 실행
    done    = set()
    results = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
            done = {r["image_id"] for r in results}
        print(f"[{model_type}] 이어서: {len(done)}개 완료")

    print(f"\n[{model_type}] DRF 평가 중... ({len(responses)}개)")
    for item in tqdm(responses):
        image_id = item["image_id"]
        if image_id in done:
            continue

        eval_result = call_gpt4o_eval(
            item["ground_truth"],
            item["response"],
        )
        if eval_result is None:
            continue

        results.append({
            "image_id":              image_id,
            "model":                 model_type,
            "faithfulness_score":    eval_result.get("overall_faithfulness_score", 0),
            "total_claims":          eval_result.get("total_claims", 0),
            "faithful_claims":       eval_result.get("faithful_claims", 0),
            "hallucination_claims":  eval_result.get("hallucination_claims", 0),
            "errors":                eval_result.get("errors", []),
            "n_differences":         item["n_differences"],
        })

        time.sleep(1)

        if len(results) % 20 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[{model_type}] 완료: {len(results)}개 → {out_path}")

# ── 최종 비교 ─────────────────────────────────────────────
print("\n=== DRF 점수 비교 ===")
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

baseline_scores  = []
finetuned_scores = []

with open("results/spd_bench/baseline_eval.json") as f:
    b_data = {r["image_id"]: r for r in json.load(f)}
with open("results/spd_bench/finetuned_eval.json") as f:
    ft_data = {r["image_id"]: r for r in json.load(f)}

common_ids = list(set(b_data.keys()) & set(ft_data.keys()))
for id in common_ids:
    baseline_scores.append(b_data[id]["faithfulness_score"])
    finetuned_scores.append(ft_data[id]["faithfulness_score"])

b_mean  = np.mean(baseline_scores)
ft_mean = np.mean(finetuned_scores)
t_stat, t_p   = ttest_rel(finetuned_scores, baseline_scores)
w_stat, w_p   = wilcoxon(finetuned_scores, baseline_scores)

print(f"샘플 수:            {len(common_ids)}개")
print(f"Baseline DRF:       {b_mean:.4f}")
print(f"Fine-tuned DRF:     {ft_mean:.4f}")
print(f"개선폭:             {ft_mean - b_mean:+.4f}")
print(f"t-test p-value:     {t_p:.4f}")
print(f"Wilcoxon p-value:   {w_p:.4f}")

if ft_mean > b_mean and t_p < 0.05:
    print("\n✅ 통계적으로 유의미한 faithfulness 향상!")
    print("→ 'VF-Score 필터링이 모델 faithfulness를 높인다' 결론 도출")
elif ft_mean > b_mean:
    print("\n⚠️ 향상됐지만 통계적으로 유의미하지 않음 (샘플 수 부족 가능)")
else:
    print("\n❌ 향상되지 않음 — 파인튜닝 전략 재검토 필요")

# 저장
summary = {
    "n_samples":       len(common_ids),
    "baseline_drf":    round(b_mean, 4),
    "finetuned_drf":   round(ft_mean, 4),
    "improvement":     round(ft_mean - b_mean, 4),
    "ttest_p":         round(t_p, 4),
    "wilcoxon_p":      round(w_p, 4),
}
with open("results/spd_bench/final_comparison.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n저장 완료 → results/spd_bench/final_comparison.json")