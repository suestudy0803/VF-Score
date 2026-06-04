import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from PIL import Image, ImageDraw
import requests
from io import BytesIO

MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"
processor = LlavaNextProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),  # VRAM 절약
)

# 테스트 이미지 생성 (간단한 컬러 블록)
image = Image.new("RGB", (336, 336), color=(100, 149, 237))
draw = ImageDraw.Draw(image)
draw.rectangle([50, 50, 286, 286], fill=(255, 165, 0), outline=(0, 0, 0), width=3)

prompt = """[INST] <image>
Look at this image carefully and answer step by step:
What animal is in the image? What is it doing?
Think through each step before giving your final answer. [/INST]"""

inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=300, do_sample=False)

print(processor.decode(output[0], skip_special_tokens=True))