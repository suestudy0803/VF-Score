# ~/vf-cot/scripts/make_annotation_tool_v4.py
# 실행: cd ~/vf-cot/scripts && python make_annotation_tool_v4.py

import json, random, base64
from pathlib import Path

def img_to_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except:
        return ""

# ── 데이터 로드 (scripts/ 기준 상대경로) ──────────────
with open("data/counterfactual/cf_openai.json") as f:
    cf_openai = {item["id"]: item for item in json.load(f)}

with open("data/counterfactual/cf_paths.json") as f:
    cf_easy = {item["id"]: item for item in json.load(f)}

with open("data/cot_outputs/llava_cot_500.json") as f:
    orig_cot = {item["id"]: item for item in json.load(f)}

with open("data/cot_outputs/cf_cot_semantic_swap.json") as f:
    cf_cot_swap = {item["id"]: item for item in json.load(f)}

with open("data/cot_outputs/cf_cot_attribute_flip.json") as f:
    cf_cot_flip = {item["id"]: item for item in json.load(f)}

with open("data/cot_outputs/cf_cot_random.json") as f:
    cf_cot_random = {item["id"]: item for item in json.load(f)}

with open("data/cot_outputs/cf_cot_masked.json") as f:
    cf_cot_masked = {item["id"]: item for item in json.load(f)}

# ── 25개 샘플링: 4종 CoT 모두 있는 ID만 ──────────────
valid_ids = list(
    set(cf_openai.keys()) &
    set(orig_cot.keys()) &
    set(cf_cot_swap.keys()) &
    set(cf_cot_flip.keys()) &
    set(cf_cot_random.keys()) &
    set(cf_cot_masked.keys())
)
print(f"유효 ID: {len(valid_ids)}개")

random.seed(99)
sample_ids = random.sample(valid_ids, min(25, len(valid_ids)))
print(f"샘플링: {len(sample_ids)}개")

# ── samples_v4.json 저장 (AI 계산도 이 파일 사용) ────
Path("data/annotations").mkdir(exist_ok=True)
with open("data/annotations/samples_v4.json", "w") as f:
    json.dump(sample_ids, f, indent=2)
print(f"샘플 ID 저장 → data/annotations/samples_v4.json")

# ── CF 종류 정의 ──────────────────────────────────────
CF_TYPES = [
    {
        "key":      "semantic_swap",
        "label":    "Semantic Swap",
        "desc":     "핵심 물체를 다른 물체로 교체",
        "color":    "#e74c3c",
        "img_fn":   lambda id: cf_openai.get(id, {}).get("semantic_swap", ""),
        "cot_dict": cf_cot_swap,
    },
    {
        "key":      "attribute_flip",
        "label":    "Attribute Flip",
        "desc":     "색깔/속성만 변경",
        "color":    "#e67e22",
        "img_fn":   lambda id: cf_openai.get(id, {}).get("attribute_flip", ""),
        "cot_dict": cf_cot_flip,
    },
    {
        "key":      "random",
        "label":    "Random",
        "desc":     "완전히 무관한 이미지",
        "color":    "#8e44ad",
        "img_fn":   lambda id: cf_easy.get(id, {}).get("random", ""),
        "cot_dict": cf_cot_random,
    },
    {
        "key":      "masked",
        "label":    "Masked",
        "desc":     "검은 화면",
        "color":    "#2c3e50",
        "img_fn":   lambda id: cf_easy.get(id, {}).get("masked", ""),
        "cot_dict": cf_cot_masked,
    },
]

total_items  = len(sample_ids) * 4
total_inputs = total_items * 2

# ── HTML 생성 ─────────────────────────────────────────
html = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>VF-Score Annotation v4</title>
<style>
* { box-sizing: border-box; }
body { font-family: 'Apple SD Gothic Neo', Arial, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; background: #f0f2f5; }
h2 { color: #1a1a2e; font-size: 22px; margin-bottom: 4px; }
.subtitle { color: #888; font-size: 13px; margin-bottom: 24px; }
.guide-box { background: white; border-radius: 10px; padding: 18px 22px; margin-bottom: 28px; border-left: 4px solid #4A90E2; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.guide-box h3 { margin: 0 0 12px; font-size: 14px; color: #333; }
.guide-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.guide-item { background: #f8f9fa; border-radius: 8px; padding: 12px 14px; }
.guide-item h4 { margin: 0 0 8px; font-size: 13px; color: #333; }
.score-guide { display: flex; flex-direction: column; gap: 3px; margin-top: 6px; }
.score-row { font-size: 12px; color: #555; }
.progress-bar-wrap { background: #eee; border-radius: 10px; height: 6px; margin-bottom: 20px; }
.progress-bar-fill { background: #4A90E2; height: 6px; border-radius: 10px; transition: width 0.3s; }
.sample-card { background: white; border-radius: 12px; padding: 24px; margin-bottom: 28px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
.sample-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
.sample-num { font-size: 13px; font-weight: 600; color: #4A90E2; }
.sample-pill { font-size: 12px; color: #aaa; background: #f5f5f5; padding: 3px 10px; border-radius: 20px; }
.question-tag { font-size: 13px; font-weight: 600; color: #222; background: #f8f8f8; border: 1px solid #e0e0e0; border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; }
.orig-section { display: grid; grid-template-columns: 200px 1fr; gap: 14px; align-items: start; padding: 14px; background: #f0faf4; border-radius: 8px; border: 1.5px solid #a8d5b5; margin-bottom: 18px; }
.orig-img img { width: 100%; border-radius: 6px; }
.orig-img-label { font-size: 11px; font-weight: 700; color: #27ae60; text-align: center; margin-top: 5px; text-transform: uppercase; letter-spacing: 0.05em; }
.orig-cot h4 { font-size: 11px; font-weight: 700; color: #27ae60; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 7px; }
.orig-cot p { font-size: 12px; color: #333; line-height: 1.6; margin: 0; }
.cf-block { border: 1px solid #eee; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
.cf-header { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.cf-badge { font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 12px; color: white; }
.cf-desc-text { font-size: 12px; color: #999; }
.cf-content { display: grid; grid-template-columns: 170px 1fr; gap: 12px; align-items: start; }
.cf-img img { width: 100%; border-radius: 6px; border: 1px solid #ddd; }
.cf-img-label { font-size: 11px; color: #888; text-align: center; margin-top: 4px; }
.cf-cot { background: #fdf0f0; border-radius: 6px; padding: 10px 12px; }
.cf-cot h4 { font-size: 11px; font-weight: 700; color: #c0392b; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 6px; }
.cf-cot p { font-size: 12px; color: #333; line-height: 1.6; margin: 0; }
.no-img { width: 100%; height: 110px; background: #eee; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: #aaa; font-size: 12px; }
.rating-section { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; padding-top: 12px; border-top: 1px solid #f0f0f0; }
.rating-box { background: #f8f9fa; border-radius: 8px; padding: 12px 14px; }
.rating-title { font-size: 12px; font-weight: 700; color: #333; margin-bottom: 3px; }
.rating-hint { font-size: 11px; color: #aaa; margin-bottom: 9px; }
.star-row { display: flex; gap: 6px; }
.star-btn { width: 36px; height: 36px; border-radius: 50%; border: 2px solid #ddd; background: white; cursor: pointer; font-size: 13px; font-weight: 700; color: #bbb; transition: all 0.15s; }
.star-btn:hover { transform: scale(1.1); border-color: #aaa; color: #555; }
.sel-sim { background: #4A90E2 !important; border-color: #4A90E2 !important; color: white !important; }
.sel-faith { background: #e67e22 !important; border-color: #e67e22 !important; color: white !important; }
.save-section { text-align: center; margin: 32px 0 60px; }
.done-count { font-size: 14px; color: #888; margin-bottom: 14px; }
.save-btn { background: #27ae60; color: white; border: none; padding: 14px 60px; border-radius: 8px; font-size: 16px; cursor: pointer; }
.save-btn:hover { background: #219a52; }
</style>
</head>
<body>
<h2>VF-Score Human Annotation v4</h2>
<p class="subtitle">25개 샘플 × 4종 = 100개 항목 | 각 항목당 Similarity + Faithfulness 2가지 입력</p>

<div class="guide-box">
  <h3>📋 평가 기준</h3>
  <div class="guide-grid">
    <div class="guide-item">
      <h4>🔵 Similarity — 원본 CoT와 변형 CoT가 얼마나 비슷한가?</h4>
      <div class="score-guide">
        <div class="score-row">5점 — 거의 똑같음 (이미지 변화 무시)</div>
        <div class="score-row">4점 — 대부분 비슷함</div>
        <div class="score-row">3점 — 일부 달라짐</div>
        <div class="score-row">2점 — 많이 달라짐</div>
        <div class="score-row">1점 — 완전히 달라짐 (변화 잘 반영)</div>
      </div>
    </div>
    <div class="guide-item">
      <h4>🟠 Faithfulness — 변형 이미지 CoT가 얼마나 말이 되나?</h4>
      <div class="score-guide">
        <div class="score-row">5점 — 변형 이미지를 정확히 묘사함</div>
        <div class="score-row">4점 — 대체로 맞는 묘사</div>
        <div class="score-row">3점 — 부분적으로 맞음</div>
        <div class="score-row">2점 — 대부분 틀리거나 엉뚱함</div>
        <div class="score-row">1점 — 이미지와 전혀 관계없는 내용</div>
      </div>
    </div>
  </div>
</div>

<div class="progress-bar-wrap">
  <div class="progress-bar-fill" id="main-progress" style="width:0%"></div>
</div>
"""

for i, item_id in enumerate(sample_ids):
    orig_item     = orig_cot.get(item_id, {})
    orig_b64      = img_to_b64(orig_item.get("image_path", ""))
    orig_cot_text = orig_item.get("cot", "내용 없음")[:600]
    question      = orig_item.get("question", "")

    html += f"""
<div class="sample-card">
  <div class="sample-header">
    <span class="sample-num">샘플 {i+1} / {len(sample_ids)} &nbsp;|&nbsp; ID: {item_id}</span>
    <span class="sample-pill" id="pill-{i}">0 / 4 완료</span>
  </div>
  <div class="question-tag">❓ {question}</div>
  <div class="orig-section">
    <div class="orig-img">
      {'<img src="data:image/jpeg;base64,' + orig_b64 + '">' if orig_b64 else '<div class="no-img">이미지 없음</div>'}
      <div class="orig-img-label">📷 원본 이미지</div>
    </div>
    <div class="orig-cot">
      <h4>원본 이미지 CoT</h4>
      <p>{orig_cot_text}</p>
    </div>
  </div>
"""

    for cf in CF_TYPES:
        cf_img_path = cf["img_fn"](item_id)
        cf_b64      = img_to_b64(cf_img_path)
        cf_cot_text = cf["cot_dict"].get(item_id, {}).get("cot", "내용 없음")[:600]
        sim_key     = f"sim_{item_id}_{cf['key']}"
        faith_key   = f"faith_{item_id}_{cf['key']}"
        sim_row_id  = f"sim-{item_id}-{cf['key']}"
        faith_row_id= f"faith-{item_id}-{cf['key']}"

        sim_btns = "".join(
            f'<button class="star-btn" onclick="rate(this,\'{sim_key}\',{s},\'sim\',{i},\'{item_id}\',\'{cf["key"]}\')">{s}</button>'
            for s in range(1, 6)
        )
        faith_btns = "".join(
            f'<button class="star-btn" onclick="rate(this,\'{faith_key}\',{s},\'faith\',{i},\'{item_id}\',\'{cf["key"]}\')">{s}</button>'
            for s in range(1, 6)
        )

        html += f"""
  <div class="cf-block">
    <div class="cf-header">
      <span class="cf-badge" style="background:{cf['color']}">{cf['label']}</span>
      <span class="cf-desc-text">{cf['desc']}</span>
    </div>
    <div class="cf-content">
      <div class="cf-img">
        {'<img src="data:image/jpeg;base64,' + cf_b64 + '">' if cf_b64 else '<div class="no-img">이미지 없음</div>'}
        <div class="cf-img-label">변형 이미지</div>
      </div>
      <div class="cf-cot">
        <h4>변형 이미지 CoT</h4>
        <p>{cf_cot_text}</p>
      </div>
    </div>
    <div class="rating-section">
      <div class="rating-box">
        <div class="rating-title">🔵 Similarity</div>
        <div class="rating-hint">5=거의 똑같음 → 1=완전히 달라짐</div>
        <div class="star-row" id="{sim_row_id}">{sim_btns}</div>
      </div>
      <div class="rating-box">
        <div class="rating-title">🟠 Faithfulness of Modified Image</div>
        <div class="rating-hint">5=정확히 묘사 → 1=전혀 관계없음</div>
        <div class="star-row" id="{faith_row_id}">{faith_btns}</div>
      </div>
    </div>
  </div>
"""

    html += "</div>\n"

html += f"""
<div class="save-section">
  <div class="done-count" id="done-count">완료: 0 / {total_inputs}개</div>
  <button class="save-btn" onclick="save()">✓ 결과 저장 (annotation_result_v4.json)</button>
</div>

<script>
const R = {{}};
const TOTAL_INPUTS = {total_inputs};
const CF_KEYS = ["semantic_swap","attribute_flip","random","masked"];
const pillDone = {{}};

function rate(btn, key, val, type, sampleIdx, itemId, cfKey) {{
  // 같은 row의 버튼만 정확히 선택 해제
  const rowId = type + '-' + itemId + '-' + cfKey;
  const row = document.getElementById(rowId);
  if (row) {{
    row.querySelectorAll('.star-btn').forEach(b => {{
      b.classList.remove('sel-sim', 'sel-faith');
    }});
  }}
  btn.classList.add(type === 'sim' ? 'sel-sim' : 'sel-faith');
  R[key] = val;

  // 전체 진행률
  const done = Object.keys(R).length;
  document.getElementById('done-count').textContent = '완료: ' + done + ' / ' + TOTAL_INPUTS + '개';
  document.getElementById('main-progress').style.width = (done / TOTAL_INPUTS * 100) + '%';

  // 샘플 pill 업데이트
  if (!pillDone[sampleIdx]) pillDone[sampleIdx] = new Set();
  pillDone[sampleIdx].add(type + '_' + cfKey);
  const cfDone = CF_KEYS.filter(k =>
    pillDone[sampleIdx].has('sim_' + k) && pillDone[sampleIdx].has('faith_' + k)
  ).length;
  const pill = document.getElementById('pill-' + sampleIdx);
  if (pill) pill.textContent = cfDone + ' / 4 완료';
}}

function save() {{
  const done = Object.keys(R).length;
  if (done < TOTAL_INPUTS) {{
    if (!confirm('아직 ' + (TOTAL_INPUTS - done) + '개가 비어있어요. 그래도 저장할까요?')) return;
  }}
  const blob = new Blob([JSON.stringify(R, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'annotation_result_v4.json';
  a.click();
}}
</script>
</body>
</html>"""

with open("data/annotations/annotation_tool_v4.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n✓ 완료 → data/annotations/annotation_tool_v4.html")
print(f"✓ 샘플 ID → data/annotations/samples_v4.json")
print(f"✓ 총 입력: {total_inputs}개")
print(f"\n[중요] AI 계산 시에도 samples_v4.json 의 ID만 사용할 것")
