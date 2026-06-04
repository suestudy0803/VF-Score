# ~/vf-cot/scripts/method2_compute_vfscore.py
# 실행: cd ~/vf-cot/scripts && python method2_compute_vfscore.py

import json, os, time, base64
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
import numpy as np

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
ALPHA  = 0.05

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

질문: 이미지에서 무엇이 보이는가?
CoT A (원본): {cot_img1}
CoT B (변형): {cot_img2}

1에서 5 사이의 숫자 하나만 출력하세요."""

FAITHFULNESS_PROMPT = """당신은 주어진 이미지와 Chain-of-Thought(CoT) 추론의 충실도를 평가하는 전문 평가자입니다.

위 이미지를 직접 보고, 아래 CoT가 이 이미지를 얼마나 정확하고 논리적으로 묘사하는지 평가하세요.

점수 기준 (1~5점):
1점 — CoT가 이미지와 전혀 관계없는 내용
2점 — 대부분 이미지와 일치하지 않음
3점 — 부분적으로 맞음
4점 — 대체로 정확하게 묘사
5점 — 이미지를 정확히 묘사하고 논리적으로 추론

CoT: {cot}

1에서 5 사이의 숫자 하나만 출력하세요."""

def img_to_b64(pil_image):
    import io
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def call_gpt4o_text(prompt, max_retries=3):
    import re
    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5, temperature=0,
            )
            match = re.search(r'[1-5]', res.choices[0].message.content)
            return int(match.group()) if match else None
        except Exception as e:
            print(f"  재시도 {attempt+1}: {e}")
            time.sleep(3)
    return None

def call_gpt4o_vision(prompt, img_b64, max_retries=3):
    import re
    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                        "detail": "low"
                    }},
                    {"type": "text", "text": prompt}
                ]}],
                max_tokens=5, temperature=0,
            )
            match = re.search(r'[1-5]', res.choices[0].message.content)
            return int(match.group()) if match else None
        except Exception as e:
            print(f"  재시도 {attempt+1}: {e}")
            time.sleep(3)
    return None

def normalize(score):
    return (score - 1) / 4

def compute_vf_score(sim, faith, alpha=ALPHA):
    return alpha * (1 - normalize(sim)) + (1 - alpha) * normalize(faith)

# ── 데이터 로드 ───────────────────────────────────────────
with open("data/spd_finetune/cot_pairs.json") as f:
    cot_pairs = {r["image_id"]: r for r in json.load(f)}
print(f"CoT 쌍: {len(cot_pairs)}개")

# 이미지 로드 (VF-Score 계산에 image2 필요)
print("SPD medium 로드 중...")
ds = load_dataset("Jackson-Lv/SPD-Faith-Bench", split="medium")
ds_dict = {str(item["image_id"]): item for item in ds}

# ── 이어서 실행 ───────────────────────────────────────────
Path("data/spd_finetune").mkdir(parents=True, exist_ok=True)
out_path = "data/spd_finetune/vf_scores.json"
done = set()
results = []
if os.path.exists(out_path):
    with open(out_path) as f:
        results = json.load(f)
        done = {r["image_id"] for r in results}
    print(f"이어서 실행: {len(done)}개 완료")

print(f"VF-Score 계산 중... ({len(cot_pairs)}개)")
for image_id, pair in tqdm(cot_pairs.items()):
    if image_id in done:
        continue

    cot1 = pair["cot_img1"]
    cot2 = pair["cot_img2"]

    # Similarity (텍스트만)
    sim_prompt = SIMILARITY_PROMPT.format(
        cot_img1=cot1[:600], cot_img2=cot2[:600]
    )
    sim_score = call_gpt4o_text(sim_prompt)
    time.sleep(1)

    # Faithfulness (image2 + CoT2)
    ds_item = ds_dict.get(image_id)
    if ds_item is None:
        continue
    img2_b64 = img_to_b64(ds_item["image2"])
    faith_prompt = FAITHFULNESS_PROMPT.format(cot=cot2[:600])
    faith_score = call_gpt4o_vision(faith_prompt, img2_b64)
    time.sleep(1)

    if sim_score is None or faith_score is None:
        continue

    vf = compute_vf_score(sim_score, faith_score)
    results.append({
        "image_id":    image_id,
        "sim_score":   sim_score,
        "faith_score": faith_score,
        "vf_score":    round(vf, 4),
    })

    if len(results) % 30 == 0:
        with open(out_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  {len(results)}개 저장")

with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

scores = [r["vf_score"] for r in results]
print(f"\n완료: {len(results)}개")
print(f"평균 VF-Score: {np.mean(scores):.4f}")
print(f"중간값:        {np.median(scores):.4f}")
threshold = float(np.median(scores))
high = [r for r in results if r["vf_score"] >= threshold]
print(f"High VF-Score (>= {threshold:.3f}): {len(high)}개")

# High VF-Score ID 저장
with open("data/spd_finetune/high_vf_ids.json", "w") as f:
    json.dump([r["image_id"] for r in high], f, indent=2)
print(f"저장 → data/spd_finetune/high_vf_ids.json")