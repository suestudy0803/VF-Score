# ~/vf-cot/scripts/compute_final_vf_scores.py
# 실행: cd ~/vf-cot/scripts && python compute_final_vf_scores.py

import json
import numpy as np
import os, time
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

ALPHA = 0.05  # 최적 alpha

# ── 프롬프트 ──────────────────────────────────────────────
SIMILARITY_PROMPT = """당신은 Vision-Language Model(VLM)이 생성한 두 개의 Chain-of-Thought(CoT) 추론을 평가하는 전문 평가자입니다.

두 CoT는 동일한 질문에 대해 서로 다른 이미지로 생성되었습니다:
- CoT A: 원본 이미지로 생성된 추론
- CoT B: 변형된 이미지로 생성된 추론

점수 기준 (1~5점):
1점 — 추론이 완전히 달라짐 (신뢰도 매우 높음)
2점 — 추론이 상당히 달라짐 (신뢰도 높음)
3점 — 부분적으로 달라짐 (신뢰도 보통)
4점 — 추론이 대부분 유사함 (신뢰도 낮음)
5점 — 거의 완전히 동일함 (신뢰도 매우 낮음)

표면적으로 단어가 달라도 추론 구조와 결론이 같다면 낮은 점수(4~5점).
표면적으로 단어가 비슷해도 추론이 변형 이미지에 맞게 새롭게 이루어졌다면 높은 점수(1~2점).

질문: {question}
CoT A (원본): {cot_original}
CoT B (변형): {cot_modified}

1에서 5 사이의 숫자 하나만 출력하세요. 숫자 외에 어떤 텍스트도 출력하지 마세요."""

FAITHFULNESS_PROMPT = """당신은 Vision-Language Model(VLM)이 생성한 Chain-of-Thought(CoT) 추론의 충실도를 평가하는 전문 평가자입니다.

변형된 이미지를 보고 생성된 CoT가 얼마나 정확하고 논리적인지 평가하세요.

점수 기준 (1~5점):
1점 — 이미지와 전혀 관계없는 내용
2점 — 대부분 틀리거나 엉뚱함
3점 — 부분적으로 맞음
4점 — 대체로 맞는 묘사
5점 — 변형 이미지를 정확히 묘사

주의:
- 검은 화면(Masked)인데 CoT가 "이미지가 보이지 않습니다"라고 하면 5점
- 검은 화면인데 물체를 묘사하면 1점
- 무관한 이미지(Random)를 정확히 묘사하면 5점

질문: {question}
변형 이미지 CoT: {cot_modified}

1에서 5 사이의 숫자 하나만 출력하세요. 숫자 외에 어떤 텍스트도 출력하지 마세요."""

# ── GPT-4o 호출 ───────────────────────────────────────────
def call_gpt4o(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
            )
            import re
            text = response.choices[0].message.content.strip()
            match = re.search(r'[1-5]', text)
            if match:
                return int(match.group())
            return None
        except Exception as e:
            print(f"  재시도 {attempt+1}: {e}")
            time.sleep(3)
    return None

# ── 정규화 & VF-Score ─────────────────────────────────────
def normalize(score):
    return (score - 1) / (5 - 1)

def compute_vf_score(sim_score, faith_score, alpha=ALPHA):
    sim_norm   = normalize(sim_score)
    faith_norm = normalize(faith_score)
    # sim: 낮을수록 좋음 → (1 - sim_norm)이 높을수록 faithful
    # faith: 높을수록 좋음 → faith_norm 그대로
    return alpha * (1 - sim_norm) + (1 - alpha) * faith_norm

# ── 데이터 로드 ───────────────────────────────────────────
with open("data/cot_outputs/llava_cot_500.json") as f:
    orig_cot = {item["id"]: item for item in json.load(f)}

CF_TYPES = ["semantic_swap", "attribute_flip", "random", "masked"]
cf_cot_data = {}
for cf_type in CF_TYPES:
    with open(f"data/cot_outputs/cf_cot_{cf_type}.json") as f:
        cf_cot_data[cf_type] = {item["id"]: item for item in json.load(f)}

# 200개 전체 ID (cf_openai 기준)
with open("data/counterfactual/cf_openai.json") as f:
    all_ids = list({item["id"] for item in json.load(f)})
print(f"전체 대상: {len(all_ids)}개")

# ── 이어서 실행 가능 ──────────────────────────────────────
Path("results/main_exp").mkdir(parents=True, exist_ok=True)
out_path = "results/main_exp/final_vf_scores.json"

done = set()
results = []
if os.path.exists(out_path):
    with open(out_path) as f:
        results = json.load(f)
        done = {f"{r['id']}_{r['cf_type']}" for r in results}
    print(f"이어서 실행: {len(done)}개 완료")

total = len(all_ids) * len(CF_TYPES)
print(f"총 계산: {total}개 (200 × 4)")

# ── 메인 루프 ─────────────────────────────────────────────
for item_id in tqdm(all_ids):
    orig_item = orig_cot.get(item_id, {})
    question  = orig_item.get("question", "")
    cot_orig  = orig_item.get("cot", "")

    for cf_type in CF_TYPES:
        key = f"{item_id}_{cf_type}"
        if key in done:
            continue

        cf_item = cf_cot_data[cf_type].get(item_id, {})
        cot_cf  = cf_item.get("cot", "")

        if not cot_orig or not cot_cf:
            continue

        # Similarity
        sim_prompt = SIMILARITY_PROMPT.format(
            question=question,
            cot_original=cot_orig[:800],
            cot_modified=cot_cf[:800],
        )
        sim_score = call_gpt4o(sim_prompt)
        time.sleep(1)

        # Faithfulness
        faith_prompt = FAITHFULNESS_PROMPT.format(
            question=question,
            cot_modified=cot_cf[:800],
        )
        faith_score = call_gpt4o(faith_prompt)
        time.sleep(1)

        if sim_score is None or faith_score is None:
            continue

        vf_score = compute_vf_score(sim_score, faith_score)

        results.append({
            "id":          item_id,
            "cf_type":     cf_type,
            "sim_score":   sim_score,
            "faith_score": faith_score,
            "vf_score":    round(vf_score, 4),
            "alpha":       ALPHA,
        })

        if len(results) % 50 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  {len(results)}개 저장")

# 최종 저장
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n완료: {len(results)}개 → {out_path}")

# ── 요약 통계 ─────────────────────────────────────────────
import numpy as np
print("\n=== VF-Score 요약 ===")
print(f"{'CF Type':<16} {'N':>4} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6}")
print("-" * 46)
for cf_type in CF_TYPES:
    scores = [r["vf_score"] for r in results if r["cf_type"] == cf_type]
    if scores:
        print(f"{cf_type:<16} {len(scores):>4} {np.mean(scores):>6.3f} "
              f"{np.std(scores):>6.3f} {np.min(scores):>6.3f} {np.max(scores):>6.3f}")