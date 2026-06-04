# ~/vf-cot/scripts/make_cf_dalle.py
import os, json, base64, time, requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SWAP_PROMPT = (
    "Edit this image by replacing the main object with a completely "
    "different object of a different category. Keep the background, "
    "lighting, and composition identical. Make it photorealistic."
)

FLIP_PROMPT = (
    "Edit this image by changing only the color of the main object "
    "to a clearly different color. Keep everything else — shape, "
    "background, position — exactly the same. Make it photorealistic."
)

def image_to_b64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def edit_image(image_path, prompt, size="1024x1024"):
    b64 = image_to_b64(image_path)

    # Step 1: GPT-4o로 편집 프롬프트 생성
    analysis = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                },
                {
                    "type": "text",
                    "text": (
                        f"{prompt}\n\n"
                        "Describe in detail what the edited image should look like, "
                        "as an image generation prompt. Be specific about objects, "
                        "colors, composition, and style. Output only the prompt."
                    )
                }
            ]
        }],
        max_tokens=300,
    )
    dalle_prompt = analysis.choices[0].message.content.strip()

    # Step 2: gpt-image-1로 생성 (base64 반환)
    response = client.images.generate(
        model="gpt-image-1",
        prompt=dalle_prompt,
        size=size,
        quality="medium",
        n=1,
    )

    # gpt-image-1은 base64로 반환
    img_bytes = base64.b64decode(response.data[0].b64_json)
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    return image, dalle_prompt

# ── 메인 루프 ─────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "data/raw/subset_500.json")) as f:
    data = json.load(f)

data = data[:200]  # 예산 $20 이내

os.makedirs(os.path.join(BASE, "data/counterfactual/semantic_swap"), exist_ok=True)
os.makedirs(os.path.join(BASE, "data/counterfactual/attribute_flip"), exist_ok=True)

done = set()
results = []
out_path = os.path.join(BASE, "data/counterfactual/cf_openai.json")
if os.path.exists(out_path):
    with open(out_path) as f:
        results = json.load(f)
        done = {r["id"] for r in results}
    print(f"이어서 실행: {len(done)}개 완료")

failed = []

for item in tqdm(data):
    item_id = item["id"]
    if item_id in done:
        continue

    try:
        image_path = item["image_path"]

        # ① Semantic Swap
        swapped, swap_prompt = edit_image(image_path, SWAP_PROMPT)
        swap_path = os.path.join(BASE, f"data/counterfactual/semantic_swap/{item_id}.jpg")
        swapped.save(swap_path)

        time.sleep(12)  # 5 images/min 제한 맞춤 (60초 / 5 = 12초)

        # ② Attribute Flip
        flipped, flip_prompt = edit_image(image_path, FLIP_PROMPT)
        flip_path = os.path.join(BASE, f"data/counterfactual/attribute_flip/{item_id}.jpg")
        flipped.save(flip_path)

        results.append({
            "id": item_id,
            "semantic_swap": swap_path,
            "swap_prompt_used": swap_prompt,
            "attribute_flip": flip_path,
            "flip_prompt_used": flip_prompt,
        })

        time.sleep(12)

    except Exception as e:
        print(f"실패 [{item_id}]: {e}")
        failed.append({"id": item_id, "error": str(e)})

    if len(results) % 20 == 0 and results:
        with open(out_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"{len(results)}개 저장")

with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
with open(os.path.join(BASE, "data/counterfactual/cf_openai_failed.json"), "w") as f:
    json.dump(failed, f, indent=2)

print(f"\n완료: {len(results)}개 성공 / {len(failed)}개 실패")