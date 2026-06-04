# ~/vf-cot/scripts/make_cf_inpaint.py
import torch, json, os
from pathlib import Path
from PIL import Image
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm

# 프로젝트 루트 = 이 스크립트의 상위 디렉토리
ROOT = Path(__file__).resolve().parent.parent

# ── 모델 로드 ──────────────────────────────────────────
# SAM (물체 영역 자동 감지)
sam = sam_model_registry["vit_h"](
    checkpoint=str(ROOT / "models/sam/sam_vit_h.pth")
).to("cuda")
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=16,        # 속도 우선
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
)

# Stable Diffusion Inpainting
# 로컬 캐시 우선, 없으면 Hub에서 다운로드 (huggingface-cli login 필요)
_SD_LOCAL = ROOT / "models/checkpoints/sd2-inpainting"
_SD_MODEL = str(_SD_LOCAL) if _SD_LOCAL.exists() else "stabilityai/stable-diffusion-2-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    _SD_MODEL,
    torch_dtype=torch.bfloat16,   # RTX 5090
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()

# ── 핵심 함수 ──────────────────────────────────────────
def get_largest_mask(image_np):
    """SAM으로 이미지에서 가장 큰 물체 마스크 추출"""
    masks = mask_generator.generate(image_np)
    if not masks:
        return None
    # 가장 큰 영역 선택
    largest = max(masks, key=lambda m: m["area"])
    mask = largest["segmentation"].astype(np.uint8) * 255
    return Image.fromarray(mask)

def inpaint(image, mask, prompt, negative_prompt="blurry, low quality"):
    """마스킹된 영역을 prompt 내용으로 채우기"""
    image = image.resize((512, 512))
    mask  = mask.resize((512, 512))
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    return result.resize(image.size)

# ── 교체 프롬프트 정의 ─────────────────────────────────
# subject별로 Semantic Swap 프롬프트 설정
SWAP_PROMPTS = {
    "natural science": "a different scientific apparatus, photorealistic",
    "social science":  "a different everyday object, photorealistic",
    "language science":"a different text diagram, photorealistic",
}
DEFAULT_SWAP = "a completely different object, photorealistic"

FLIP_PROMPTS = {
    "natural science": "the same object but different color, photorealistic",
    "social science":  "the same object but different color, photorealistic",
    "language science":"the same diagram but different color scheme",
}

# ── 메인 루프 ─────────────────────────────────────────
with open(ROOT / "data/raw/subset_500.json") as f:
    data = json.load(f)

os.makedirs(ROOT / "data/counterfactual/semantic_swap", exist_ok=True)
os.makedirs(ROOT / "data/counterfactual/attribute_flip", exist_ok=True)

results = []
failed = []

for item in tqdm(data):
    item_id = item["id"]
    subject = item.get("subject", "natural science")

    try:
        image = Image.open(item["image_path"]).convert("RGB")
        image_np = np.array(image)

        # SAM으로 마스크 생성
        mask = get_largest_mask(image_np)
        if mask is None:
            failed.append(item_id)
            continue

        # ① Semantic Swap
        swap_prompt = SWAP_PROMPTS.get(subject, DEFAULT_SWAP)
        swapped = inpaint(image, mask, swap_prompt)
        swap_path = str(ROOT / f"data/counterfactual/semantic_swap/{item_id}.jpg")
        swapped.save(swap_path)

        # ② Attribute Flip
        flip_prompt = FLIP_PROMPTS.get(subject, DEFAULT_SWAP)
        flipped = inpaint(image, mask, flip_prompt)
        flip_path = str(ROOT / f"data/counterfactual/attribute_flip/{item_id}.jpg")
        flipped.save(flip_path)

        results.append({
            "id": item_id,
            "semantic_swap": swap_path,
            "attribute_flip": flip_path,
        })

    except Exception as e:
        print(f"실패 [{item_id}]: {e}")
        failed.append(item_id)

    # 50개마다 중간 저장
    if len(results) % 50 == 0:
        with open(ROOT / "data/counterfactual/cf_inpaint.json", "w") as f:
            json.dump(results, f, indent=2)

# 최종 저장
with open(ROOT / "data/counterfactual/cf_inpaint.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n완료: {len(results)}개 성공 / {len(failed)}개 실패")
print(f"실패 ID: {failed[:10]}")