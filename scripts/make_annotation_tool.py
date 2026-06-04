# ~/vf-cot/scripts/make_annotation_tool.py
import json, random, base64
from pathlib import Path

def img_to_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except:
        return ""

with open("data/cot_outputs/llava_cot_500.json") as f:
    data = json.load(f)

random.seed(42)
samples = random.sample(data, 50)

html = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>CoT Faithfulness Annotation</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
  h2 { color: #333; }
  .card { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
  .question { font-weight: bold; font-size: 15px; margin: 10px 0; color: #222; }
  .cot-step { background: #f9f9f9; padding: 12px; border-radius: 6px; margin: 8px 0; border-left: 3px solid #ddd; }
  .step-num { font-size: 11px; color: #999; margin-bottom: 4px; }
  .step-text { font-size: 14px; color: #333; line-height: 1.5; }
  .labels { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
  .label-btn { padding: 5px 12px; border-radius: 20px; border: 1.5px solid #ccc; cursor: pointer; font-size: 12px; background: white; transition: all 0.15s; }
  .label-btn:hover { background: #f0f0f0; }
  .label-btn.selected { background: #4A90E2; color: white; border-color: #4A90E2; }
  img { max-width: 280px; border-radius: 6px; margin: 10px 0; display: block; }
  .save-btn { background: #27ae60; color: white; border: none; padding: 14px 36px; border-radius: 8px; font-size: 16px; cursor: pointer; margin: 20px 0 60px; display: block; }
  .save-btn:hover { background: #219a52; }
  .progress-bar { background: #eee; border-radius: 10px; height: 6px; margin-bottom: 6px; }
  .progress-fill { background: #4A90E2; height: 6px; border-radius: 10px; }
  .progress-text { color: #888; font-size: 12px; margin-bottom: 16px; }
  .label-guide { background: #EEF4FF; border-radius: 6px; padding: 12px 16px; margin-bottom: 20px; font-size: 13px; color: #444; line-height: 1.7; }
</style>
</head>
<body>
<h2>CoT Faithfulness Annotation</h2>
<div class="label-guide">
  <strong>레이블 기준:</strong><br>
  <b>1 Visually Grounded</b> — 이미지를 직접 보고 쓴 내용 (이미지에 근거 있음)<br>
  <b>2 Plausible but Vague</b> — 그럴듯하지만 이미지 근거가 불명확함<br>
  <b>3 Language Prior Only</b> — 이미지 없이 언어 패턴으로만 쓴 내용<br>
  <b>4 Hallucinated</b> — 이미지에 없는 내용을 지어낸 것
</div>
"""

for i, item in enumerate(samples):
    cot_text = item.get("cot", "")
    steps = [s.strip() for s in cot_text.split("\n") if s.strip()][:4]

    b64 = img_to_b64(item["image_path"])
    img_tag = f'<img src="data:image/jpeg;base64,{b64}" alt="image">' if b64 else "<p style='color:#999'>이미지 없음</p>"

    pct = int((i / len(samples)) * 100)
    html += f"""
<div class="card">
  <div class="progress-bar"><div class="progress-fill" style="width:{pct}%"></div></div>
  <div class="progress-text">샘플 {i+1} / {len(samples)}</div>
  {img_tag}
  <div class="question">Q: {item['question']}</div>
  <div style="font-size:13px;font-weight:500;margin:12px 0 6px;color:#555">CoT 추론 단계별 레이블링:</div>
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
<button class="save-btn" onclick="save()">✓ 결과 저장 (JSON 다운로드)</button>
<script>
const R = {};
function sel(btn, key, val) {
  btn.parentElement.querySelectorAll('.label-btn').forEach(b => b.classList.remove('selected'));
  btn.classList.add('selected');
  R[key] = val;
}
function save() {
  const total = """ + str(sum(len([s.strip() for s in item.get('cot','').split('\n') if s.strip()][:4]) for item in samples)) + """;
  const done = Object.keys(R).length;
  if (done < total) {
    if (!confirm(`아직 ${total - done}개 레이블이 비어있어요. 그래도 저장할까요?`)) return;
  }
  const blob = new Blob([JSON.stringify(R, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'annotation_result.json';
  a.click();
}
</script>
</body>
</html>"""

Path("data/annotations").mkdir(exist_ok=True)
with open("data/annotations/annotation_tool.html", "w", encoding="utf-8") as f:
    f.write(html)

with open("data/annotations/samples_50.json", "w") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"완료: {len(samples)}개 샘플 → data/annotations/annotation_tool.html")