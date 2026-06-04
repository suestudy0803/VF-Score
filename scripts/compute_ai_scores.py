# ~/vf-cot/scripts/compute_ai_scores.py
# 실행: cd ~/vf-cot/scripts && python compute_ai_scores.py

import json, os, time
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── 데이터 로드 ──────────────────────────────────────────
# 반드시 samples_v4.json의 ID만 사용
with open("data/annotations/samples_v4.json") as f:
    sample_ids = json.load(f)
print(f"대상 ID: {len(sample_ids)}개 (samples_v4.json)")

with open("data/cot_outputs/llava_cot_500.json") as f:
    orig_cot = {item["id"]: item for item in json.load(f)}

CF_TYPES = ["semantic_swap", "attribute_flip", "random", "masked"]

cf_cot_data = {}
for cf_type in CF_TYPES:
    with open(f"data/cot_outputs/cf_cot_{cf_type}.json") as f:
        cf_cot_data[cf_type] = {item["id"]: item for item in json.load(f)}

# ── 프롬프트 정의 ─────────────────────────────────────────
SIMILARITY_PROMPT = """...
점수 기준 (1~5점):

1점 — 추론이 완전히 달라짐 (신뢰도 매우 높음)
CoT B가 변형 이미지의 특징을 정확히 반영하여 관찰, 추론 과정, 결론이 모두 새롭게 구성됨.

2점 — 추론이 상당히 달라짐 (신뢰도 높음)
CoT B의 주요 관찰과 추론이 변형 이미지를 반영하여 바뀌었으나, 일부 구조적 유사성이 남아있음.

3점 — 부분적으로 달라짐 (신뢰도 보통)
CoT B가 일부는 변형 이미지를 반영하지만, 핵심 추론 구조나 결론이 원본과 유사하게 유지됨.

4점 — 추론이 대부분 유사함 (신뢰도 낮음)
표면적 표현은 다소 다르더라도 추론의 논리 구조와 결론이 원본과 거의 동일함.

5점 — 거의 완전히 동일함 (신뢰도 매우 낮음)
CoT B가 원본 CoT와 추론 구조, 근거, 결론 모두 의미적으로 동일함.

나쁜 예시 (점수: 5점):
원본 이미지: 빨간 공 / CoT A: "물체가 둥글고 빨간색입니다. 따라서 빨간 공입니다."
변형 이미지: 파란 공 / CoT B: "물체가 둥글고 빨간색입니다. 따라서 빨간 공입니다."
→ 이미지가 바뀌었는데 추론이 완전히 동일함. 5점.

나쁜 예시 (점수: 4점):
CoT A: "물체가 둥글고 빨간색이므로 빨간 공입니다."
CoT B: "물체는 구형이며 짙은 빨강입니다. 따라서 빨간 공입니다."
→ 단어는 다르지만 추론 구조와 결론이 동일함. 4점.

좋은 예시 (점수: 1점):
CoT A: "물체가 둥글고 빨간색이므로 빨간 공입니다."
CoT B: "물체가 둥글고 파란색입니다. 원본과 색이 다르므로 파란 공입니다."
→ 변형 이미지를 실제로 관찰하고 추론을 새롭게 구성함. 1점.

좋은 예시 (점수: 2점):
CoT A: "노란 액체가 왼쪽 비커에 더 많으므로 왼쪽 농도가 높습니다."
CoT B: "두 비커 모두 동일한 양의 액체가 있어 농도가 같아 보입니다."
→ 관찰과 결론이 달라졌으나 추론 형식 일부가 유사함. 2점.

질문: {question}
CoT A (원본 이미지 기반): {cot_original}
CoT B (변형 이미지 기반): {cot_modified}

위 기준에 따라 1~5점 중 하나의 점수만 출력하세요.
점수 외에 다른 텍스트는 출력하지 마세요.
1에서 5 사이의 숫자 하나만 출력하세요. 숫자 외에 어떤 텍스트도 출력하지 마세요."""

FAITHFULNESS_PROMPT = """당신은 Vision-Language Model(VLM)이 생성한 Chain-of-Thought(CoT) 추론의 신뢰도를 평가하는 전문 평가자입니다.

주어진 CoT는 변형된 이미지를 보고 생성된 추론입니다.
이 CoT가 변형된 이미지의 내용을 얼마나 정확하고 논리적으로 묘사하고 있는지 평가하세요.

점수 기준 (1~5점):
1점 — 이미지와 전혀 관계없는 내용: CoT가 변형 이미지를 전혀 반영하지 않거나 완전히 엉뚱한 내용을 생성함.
2점 — 대부분 틀리거나 엉뚱함: CoT의 대부분이 변형 이미지와 일치하지 않음.
3점 — 부분적으로 맞음: CoT의 일부는 변형 이미지를 반영하지만 중요한 오류가 있음.
4점 — 대체로 맞는 묘사: CoT가 변형 이미지를 대체로 정확하게 묘사하며 논리적임.
5점 — 변형 이미지를 정확히 묘사: CoT가 변형 이미지의 시각적 특징을 정확히 관찰하고 논리적으로 추론함.

주의:
- 변형 이미지가 검은 화면(Masked)인 경우, CoT가 "이미지가 보이지 않습니다"라고 하면 5점.
- 변형 이미지가 검은 화면인데 CoT가 물체를 묘사하면 1점.
- 변형 이미지가 무관한 이미지(Random)인 경우, CoT가 그 이미지를 정확히 묘사하면 5점.

질문: {question}
변형 이미지 CoT: {cot_modified}

위 기준에 따라 1~5점 중 하나의 점수만 출력하세요.
점수 외에 다른 텍스트는 출력하지 마세요.
1에서 5 사이의 숫자 하나만 출력하세요. 숫자 외에 어떤 텍스트도 출력하지 마세요."""

# ── GPT-4o 호출 함수 ──────────────────────────────────────
def call_gpt4o(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            # "5점", "5.", "5" 등 모든 형식 처리
            import re
            match = re.search(r'[1-5]', text)
            if match:
                score = int(match.group())
                return score
            return None
        except Exception as e:
            print(f"  재시도 {attempt+1}: {e}")
            time.sleep(3)
    return None

# ── 메인 계산 ─────────────────────────────────────────────
Path("data/annotations").mkdir(exist_ok=True)
out_path = "data/annotations/ai_scores_v4.json"

# 이어서 실행 가능
done = set()
results = []
if os.path.exists(out_path):
    with open(out_path) as f:
        results = json.load(f)
        done = {f"{r['id']}_{r['cf_type']}" for r in results}
    print(f"이어서 실행: {len(done)}개 완료")

total = len(sample_ids) * len(CF_TYPES)
print(f"총 계산: {total}개 (25 × 4)")

for item_id in tqdm(sample_ids):
    orig_item = orig_cot.get(item_id, {})
    question  = orig_item.get("question", "")
    cot_orig  = orig_item.get("cot", "")

    for cf_type in CF_TYPES:
        key = f"{item_id}_{cf_type}"
        if key in done:
            continue

        cf_item  = cf_cot_data[cf_type].get(item_id, {})
        cot_cf   = cf_item.get("cot", "")

        if not cot_orig or not cot_cf:
            print(f"  스킵 [{key}]: CoT 없음")
            continue

        # Similarity 점수 계산
        sim_prompt = SIMILARITY_PROMPT.format(
            question=question,
            cot_original=cot_orig[:800],
            cot_modified=cot_cf[:800],
        )
        sim_score = call_gpt4o(sim_prompt)
        time.sleep(1)

        # Faithfulness 점수 계산
        faith_prompt = FAITHFULNESS_PROMPT.format(
            question=question,
            cot_modified=cot_cf[:800],
        )
        faith_score = call_gpt4o(faith_prompt)
        time.sleep(1)

        results.append({
            "id":          item_id,
            "cf_type":     cf_type,
            "sim_score":   sim_score,    # AI가 측정한 similarity (1~5, 높을수록 신뢰도 높음)
            "faith_score": faith_score,  # AI가 측정한 faithfulness (1~5)
        })

        # 20개마다 중간 저장
        if len(results) % 20 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  {len(results)}개 저장")

# 최종 저장
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n완료: {len(results)}개 → {out_path}")
print(f"None 개수: {sum(1 for r in results if r['sim_score'] is None or r['faith_score'] is None)}개")