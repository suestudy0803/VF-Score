# ~/vf-cot/scripts/make_annotation_tool_v2.py
import json, random, base64
from pathlib import Path

def img_to_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except:
        return ""

# 데이터 로드
with open("data/cot_outputs/llava_cot_500.json") as f:
    cot_data = {item["id"]: item for item in json.load(f)}

with open("data/counterfactual/cf_openai.json") as f:
    cf_data = {item["id"]: item for item in json.load(f)}

with open("data/counterfactual/cf_paths.json") as f:
    easy_cf = {item["id"]: item for item in json.load(f)}

# cf_openai에 있는 것만 샘플링 (원본+바뀐이미지 둘 다 있는 것)
common_ids = list(set(cot_data.keys()) & set(cf_data.keys()))
random.seed(42)
sample_ids = random.sample(common_ids, min(50, len(common_ids)))
print(f"샘플 {len(sample_ids)}개 선정")

html = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>CoT Faithfulness Annotation v2</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
  h2 { color: #333; }
  .card { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
  .img-row { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px; margin: 12px 0; }
  .img-box { text-align: center; }
  .img-box img { width: 100%; border-radius: 6px; border: 1.5px solid #eee; }
  .img-label { font-size: 11px; color: #888; margin-top: 4px; font-weight: 500; }
  .img-label.original { color: #27ae60; }
  .img-label.swap { color: #e74c3c; }
  .img-label.flip { color: #e67e22; }
  .img-label.random { color: #8e44ad; }
  .img-label.masked { color: #555; }
  .question { font-weight: bold; font-size: 15px; margin: 12px 0 8px; color: #222; }
  .cot-step { background: #f9f9f9; padding: 12px; border-radius: 6px; margin: 8px 0; border-left: 3px solid #ddd; }
  .step-num { font-size: 11px; color: #999; margin-bottom: 4px; }
  .step-text { font-size: 14px; color: #333; line-height: 1.5; }
  .labels { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
  .label-btn { padding: 5px 12px; border-radius: 20px; border: 1.5px solid #ccc; cursor: pointer; font-size: 12px; background: white; }
  .label-btn:hover { background: #f0f0f0; }
  .label-btn.selected { background: #4A90E2; color: white; border-color: #4A90E2; }
  .save-btn { background: #27ae60; color: white; border: none; padding: 14px 36px; border-radius: 8px; font-size: 16px; cursor: pointer; margin: 20px 0 60px; display: block; }
  .progress-text { color: #888; font-size: 12px; margin-bottom: 16px; }
  .label-guide { background: #EEF4FF; border-radius: 6px; padding: 12px 16px; margin-bottom: 20px; font-size: 13px; color: #444; line-height: 1.7; }
  .divider { height: 1px; background: #eee; margin: 12px 0; }
</style>
</head>
<body>
<h2>CoT Faithfulness Annotation v2</h2>
<div class="label-guide">
  <strong>레이블 기준 — 원본 이미지와 CoT를 함께 보고 판단:</strong><br>
  <b>1 Visually Grounded</b> — 원본 이미지를 직접 보고 쓴 내용<br>
  <b>2 Plausible but Vague</b> — 그럴듯하지만 이미지 근거 불명확<br>
  <b>3 Language Prior Only</b> — 이미지 없이 언어 패턴으로만 쓴 내용<br>
  <b>4 Hallucinated</b> — 이미지에 없는 내용을 지어낸 것
</div>
"""

for i, item_id in enumerate(sample_ids):
    cot_item = cot_data[item_id]
    cf_item = cf_data[item_id]
    easy_item = easy_cf.get(item_id, {})

    steps = [s.strip() for s in cot_item.get("cot", "").split("\n") if s.strip()][:4]

    # 이미지 base64 변환
    orig_b64  = img_to_b64(cot_item["image_path"])
    swap_b64  = img_to_b64(cf_item.get("semantic_swap", ""))
    flip_b64  = img_to_b64(cf_item.get("attribute_flip", ""))
    rand_b64  = img_to_b64(easy_item.get("random", ""))
    mask_b64  = img_to_b64(easy_item.get("masked", ""))

    def img_tag(b64, label, cls):
        if b64:
            return f'<div class="img-box"><img src="data:image/jpeg;base64,{b64}"><div class="img-label {cls}">{label}</div></div>'
        return f'<div class="img-box"><div style="height:120px;background:#eee;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#aaa">없음</div><div class="img-label">{label}</div></div>'

    html += f"""
<div class="card">
  <div class="progress-text">샘플 {i+1} / {len(sample_ids)} &nbsp;|&nbsp; ID: {item_id}</div>
  <div class="img-row">
    {img_tag(orig_b64, "① 원본", "original")}
    {img_tag(swap_b64, "② Semantic Swap", "swap")}
    {img_tag(flip_b64, "③ Attribute Flip", "flip")}
    {img_tag(rand_b64, "④ Random", "random")}
  </div>
  <div class="divider"></div>
  <div class="question">Q: {cot_item['question']}</div>
  <div style="font-size:13px;font-weight:500;margin:12px 0 6px;color:#555">원본 이미지 기반 CoT — 각 Step 레이블링:</div>
"""
    for j, step in enumerate(steps):
        html += f"""
  <div class="cot-step">
    <div class="step-num">Step {j+1}</div>
    <div class="step-text">{step}</div>
    <div class="labels">
      <button class="label-btn" onclick="sel(this,'{i}_{j}',1)">1 Visually Grounded</button>
      <button class="label-btn" onclick="sel(this,'{i}_{j}',2)">2 Plausible but Vague</button>
      <button class="label-btn" onclick="sel(this,'{i}_{j}',3)">3 Language Prior Only</button>
      <button class="label-btn" onclick="sel(this,'{i}_{j}',4)">4 Hallucinated</button>
    </div>
  </div>
"""
    html += "</div>\n"

html += """
<button class="save-btn" onclick="save()">✓ 결과 저장 (JSON)</button>
<script>
const R = {};
function sel(btn, key, val) {
  btn.parentElement.querySelectorAll('.label-btn').forEach(b => b.classList.remove('selected'));
  btn.classList.add('selected');
  R[key] = val;
}
function save() {
  const blob = new Blob([JSON.stringify(R, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'annotation_result_v2.json';
  a.click();
}
</script>
</body>
</html>"""

Path("data/annotations").mkdir(exist_ok=True)
with open("data/annotations/annotation_tool_v2.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"완료 → data/annotations/annotation_tool_v2.html")