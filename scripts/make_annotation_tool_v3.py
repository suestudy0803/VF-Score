# ~/vf-cot/scripts/make_annotation_tool_v3.py
import json, random, base64
from pathlib import Path

def img_to_b64(path):
    if not path:
        return ""
    p = Path(path)
    if not p.is_absolute():
        p = Path("scripts") / p
    try:
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except:
        return ""

# ── 데이터 로드 ──────────────────────────────────────────
with open("scripts/data/counterfactual/cf_openai.json") as f:
    cf_openai = {item["id"]: item for item in json.load(f)}

with open("scripts/data/counterfactual/cf_paths.json") as f:
    cf_easy = {item["id"]: item for item in json.load(f)}

with open("scripts/data/cot_outputs/llava_cot_500.json") as f:
    orig_cot = {item["id"]: item for item in json.load(f)}

with open("scripts/data/cot_outputs/cf_cot_semantic_swap.json") as f:
    cf_cot_swap = {item["id"]: item for item in json.load(f)}

with open("scripts/data/cot_outputs/cf_cot_attribute_flip.json") as f:
    cf_cot_flip = {item["id"]: item for item in json.load(f)}

with open("scripts/data/cot_outputs/cf_cot_random.json") as f:
    cf_cot_random = {item["id"]: item for item in json.load(f)}

with open("scripts/data/cot_outputs/cf_cot_masked.json") as f:
    cf_cot_masked = {item["id"]: item for item in json.load(f)}

# 4종 CoT 모두 있는 ID만
valid_ids = list(
    set(cf_openai.keys()) &
    set(orig_cot.keys()) &
    set(cf_cot_swap.keys()) &
    set(cf_cot_flip.keys()) &
    set(cf_cot_random.keys()) &
    set(cf_cot_masked.keys())
)
print(f"유효 ID: {len(valid_ids)}개")

random.seed(42)
sample_ids = random.sample(valid_ids, min(25, len(valid_ids)))
print(f"샘플: {len(sample_ids)}개 x 4종 = {len(sample_ids)*4}번 판단")

# CF 종류 정의
CF_TYPES = [
    {
        "key":      "semantic_swap",
        "label":    "① Semantic Swap",
        "desc":     "핵심 물체를 다른 물체로 교체",
        "color":    "#e74c3c",
        "img_fn":   lambda id: cf_openai[id].get("semantic_swap", ""),
        "cot_dict": cf_cot_swap,
    },
    {
        "key":      "attribute_flip",
        "label":    "② Attribute Flip",
        "desc":     "색깔/속성만 변경",
        "color":    "#e67e22",
        "img_fn":   lambda id: cf_openai[id].get("attribute_flip", ""),
        "cot_dict": cf_cot_flip,
    },
    {
        "key":      "random",
        "label":    "③ Random",
        "desc":     "완전히 무관한 이미지",
        "color":    "#8e44ad",
        "img_fn":   lambda id: cf_easy.get(id, {}).get("random", ""),
        "cot_dict": cf_cot_random,
    },
    {
        "key":      "masked",
        "label":    "④ Masked",
        "desc":     "검은 화면",
        "color":    "#555",
        "img_fn":   lambda id: cf_easy.get(id, {}).get("masked", ""),
        "cot_dict": cf_cot_masked,
    },
]

# ── HTML 생성 ──────────────────────────────────────────
html = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>CoT Faithfulness Annotation v3</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
  h2 { color: #333; }
  .card { background: white; border: 1px solid #ddd; border-radius: 10px; padding: 24px; margin: 24px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
  .progress-bar { background: #eee; border-radius: 10px; height: 6px; margin-bottom: 8px; }
  .progress-fill { background: #4A90E2; height: 6px; border-radius: 10px; }
  .progress-text { color: #888; font-size: 12px; margin-bottom: 16px; }
  .question { font-weight: bold; font-size: 15px; margin: 0 0 16px; color: #222; background: #f8f8f8; padding: 10px 14px; border-radius: 6px; }
  .orig-img-box { text-align: center; margin-bottom: 16px; }
  .orig-img-box img { max-width: 280px; border-radius: 8px; border: 2px solid #27ae60; }
  .orig-img-label { font-size: 12px; font-weight: 700; color: #27ae60; margin-top: 6px; }
  .cf-section { border: 1px solid #eee; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
  .cf-header { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
  .cf-badge { font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 12px; color: white; }
  .cf-desc { font-size: 12px; color: #888; }
  .cf-content { display: grid; grid-template-columns: 200px 1fr 1fr; gap: 12px; align-items: start; }
  .cf-img img { width: 100%; border-radius: 6px; border: 1.5px solid #ddd; }
  .cot-box { border-radius: 6px; padding: 10px 12px; font-size: 12px; line-height: 1.6; }
  .cot-orig { background: #f0faf4; border: 1.5px solid #a8d5b5; }
  .cot-cf   { background: #fdf0f0; border: 1.5px solid #d5a8a8; }
  .cot-title { font-size: 10px; font-weight: 700; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }
  .cot-orig .cot-title { color: #27ae60; }
  .cot-cf   .cot-title { color: #c0392b; }
  .rating-row { display: flex; align-items: center; gap: 10px; margin-top: 12px; padding-top: 10px; border-top: 1px solid #f0f0f0; }
  .rating-label { font-size: 12px; font-weight: 600; color: #555; white-space: nowrap; }
  .labels { display: flex; gap: 8px; }
  .label-btn { padding: 6px 14px; border-radius: 20px; border: 1.5px solid #ccc; cursor: pointer; font-size: 12px; background: white; }
  .label-btn:hover { background: #f5f5f5; }
  .label-btn.sel-1 { background: #27ae60; color: white; border-color: #27ae60; }
  .label-btn.sel-2 { background: #f39c12; color: white; border-color: #f39c12; }
  .label-btn.sel-3 { background: #e74c3c; color: white; border-color: #e74c3c; }
  .guide-box { background: #EEF4FF; border-radius: 8px; padding: 14px 18px; margin-bottom: 24px; font-size: 13px; color: #333; line-height: 1.9; }
  .save-btn { background: #27ae60; color: white; border: none; padding: 14px 40px; border-radius: 8px; font-size: 16px; cursor: pointer; margin: 20px 0 60px; display: block; width: 100%; }
  .save-btn:hover { background: #219a52; }
</style>
</head>
<body>
<h2>CoT Faithfulness Annotation v3 — 25개 × 4종</h2>
<div class="guide-box">
  <strong>평가 방법:</strong> 각 샘플마다 원본 이미지와 4가지 바뀐 이미지를 비교하세요.<br>
  원본 CoT(초록)와 바뀐 이미지 CoT(빨강)를 읽고, 두 CoT가 얼마나 달라졌는지 판단하세요.<br><br>
  <b style="color:#27ae60">🟢 많이 달라짐</b> — 이미지 변화를 CoT가 잘 반영함 (이미지를 진짜 보고 추론)<br>
  <b style="color:#f39c12">🟡 조금 달라짐</b> — 일부만 반영<br>
  <b style="color:#e74c3c">🔴 거의 안 달라짐</b> — 이미지 바꿨는데 CoT가 거의 같음 (언어 편향)
</div>
"""

for i, item_id in enumerate(sample_ids):
    orig_item = orig_cot[item_id]
    orig_b64  = img_to_b64(orig_item["image_path"])
    orig_cot_text = orig_item.get("cot", "")[:500]
    pct = int((i / len(sample_ids)) * 100)

    html += f"""
<div class="card">
  <div class="progress-bar"><div class="progress-fill" style="width:{pct}%"></div></div>
  <div class="progress-text">샘플 {i+1} / {len(sample_ids)}</div>
  <div class="question">Q: {orig_item['question']}</div>

  <div class="orig-img-box">
    {'<img src="data:image/jpeg;base64,' + orig_b64 + '">' if orig_b64 else '<div style="width:280px;height:180px;background:#eee;display:inline-flex;align-items:center;justify-content:center;border-radius:8px;color:#aaa">없음</div>'}
    <div class="orig-img-label">📷 원본 이미지</div>
  </div>
"""

    for cf in CF_TYPES:
        cf_img_path = cf["img_fn"](item_id)
        cf_b64      = img_to_b64(cf_img_path)
        cf_cot_text = cf["cot_dict"].get(item_id, {}).get("cot", "없음")[:500]
        btn_key     = f"{item_id}_{cf['key']}"

        html += f"""
  <div class="cf-section">
    <div class="cf-header">
      <span class="cf-badge" style="background:{cf['color']}">{cf['label']}</span>
      <span class="cf-desc">{cf['desc']}</span>
    </div>
    <div class="cf-content">
      <div class="cf-img">
        {'<img src="data:image/jpeg;base64,' + cf_b64 + '">' if cf_b64 else '<div style="height:120px;background:#eee;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#aaa;font-size:12px">없음</div>'}
      </div>
      <div class="cot-box cot-orig">
        <div class="cot-title">원본 이미지 CoT</div>
        {orig_cot_text}
      </div>
      <div class="cot-box cot-cf">
        <div class="cot-title">바뀐 이미지 CoT</div>
        {cf_cot_text}
      </div>
    </div>
    <div class="rating-row">
      <span class="rating-label">두 CoT가 얼마나 달라졌나요?</span>
      <div class="labels">
        <button class="label-btn" onclick="sel(this,'{btn_key}',1)">🟢 많이 달라짐</button>
        <button class="label-btn" onclick="sel(this,'{btn_key}',2)">🟡 조금 달라짐</button>
        <button class="label-btn" onclick="sel(this,'{btn_key}',3)">🔴 거의 안 달라짐</button>
      </div>
    </div>
  </div>
"""

    html += "</div>\n"

html += f"""
<button class="save-btn" onclick="save()">✓ 결과 저장 (JSON 다운로드)</button>
<script>
const R = {{}};
const TOTAL = {len(sample_ids) * 4};
function sel(btn, key, val) {{
  const row = btn.closest('.rating-row');
  row.querySelectorAll('.label-btn').forEach(b => {{
    b.classList.remove('sel-1','sel-2','sel-3');
  }});
  btn.classList.add('sel-' + val);
  R[key] = val;
}}
function save() {{
  const done = Object.keys(R).length;
  if (done < TOTAL) {{
    if (!confirm(`아직 ${{TOTAL - done}}개가 비어있어요. 그래도 저장할까요?`)) return;
  }}
  const blob = new Blob([JSON.stringify(R, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'annotation_result_v3.json';
  a.click();
}}
</script>
</body>
</html>"""

Path("scripts/data/annotations").mkdir(exist_ok=True)
with open("scripts/data/annotations/annotation_tool_v3.html", "w", encoding="utf-8") as f:
    f.write(html)

with open("scripts/data/annotations/samples_v3.json", "w") as f:
    json.dump(sample_ids, f, indent=2)

print(f"\n완료 → scripts/data/annotations/annotation_tool_v3.html")
print(f"샘플 ID 저장 → scripts/data/annotations/samples_v3.json")
