# ~/vf-cot/scripts/finetune_llava.py
# 실행: cd ~/vf-cot/scripts && python finetune_llava.py

import os, json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

# ── 설정 ──────────────────────────────────────────────────
MODEL_PATH   = "llava-hf/llava-v1.6-mistral-7b-hf"
DATA_PATH    = "data/finetune/train_high_vf.json"
OUTPUT_DIR   = "models/checkpoints/llava-vf-lora"
LOGGING_DIR  = "logs/finetune"

# RTX 5090 최적화 설정
BATCH_SIZE       = 2
GRAD_ACCUM       = 8      # effective batch = 16
LEARNING_RATE    = 2e-4
NUM_EPOCHS       = 3
MAX_LENGTH       = 4096
LORA_RANK        = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(LOGGING_DIR).mkdir(parents=True, exist_ok=True)

# ── 데이터셋 클래스 ───────────────────────────────────────
class VFCoTDataset(Dataset):
    def __init__(self, data_path, processor, max_length=MAX_LENGTH):
        with open(data_path) as f:
            self.data = json.load(f)
        self.processor  = processor
        self.max_length = max_length
        print(f"데이터셋 로드: {len(self.data)}개")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        human_msg  = item["conversations"][0]["value"]
        gpt_msg    = item["conversations"][1]["value"]

        # 이미지 로드
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            image = Image.new("RGB", (336, 336), (0, 0, 0))

        # 프롬프트 구성
        prompt = f"[INST] {human_msg} [/INST] {gpt_msg}"

        # 토크나이즈
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )

        input_ids      = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values   = inputs["pixel_values"].squeeze(0)
        image_sizes    = inputs["image_sizes"].squeeze(0)

        # Labels: [INST] 부분은 -100으로 마스킹
        labels = input_ids.clone()
        inst_token = self.processor.tokenizer.encode("[/INST]", add_special_tokens=False)
        inst_end   = -1
        for i in range(len(input_ids) - len(inst_token)):
            if input_ids[i:i+len(inst_token)].tolist() == inst_token:
                inst_end = i + len(inst_token)
                break
        if inst_end > 0:
            labels[:inst_end] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "pixel_values":   pixel_values,
            "image_sizes":    image_sizes,
            "labels":         labels,
        }

# ── 커스텀 Collator (타일 수가 샘플마다 다를 수 있음) ──────
def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids_list, attention_mask_list, labels_list = [], [], []
    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len
        input_ids_list.append(torch.nn.functional.pad(b["input_ids"], (0, pad_len), value=0))
        attention_mask_list.append(torch.nn.functional.pad(b["attention_mask"], (0, pad_len), value=0))
        labels_list.append(torch.nn.functional.pad(b["labels"], (0, pad_len), value=-100))

    input_ids      = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    labels         = torch.stack(labels_list)
    image_sizes    = torch.stack([b["image_sizes"] for b in batch])

    pixel_values_list = [b["pixel_values"] for b in batch]
    max_tiles = max(pv.shape[0] for pv in pixel_values_list)
    padded = []
    for pv in pixel_values_list:
        if pv.shape[0] < max_tiles:
            pad = torch.zeros(max_tiles - pv.shape[0], *pv.shape[1:], dtype=pv.dtype)
            pv = torch.cat([pv, pad], dim=0)
        padded.append(pv)
    pixel_values = torch.stack(padded)

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "pixel_values":   pixel_values,
        "image_sizes":    image_sizes,
        "labels":         labels,
    }

# ── 모델 & 프로세서 로드 ──────────────────────────────────
print("프로세서 로드 중...")
processor = LlavaNextProcessor.from_pretrained(MODEL_PATH)
processor.tokenizer.pad_token = processor.tokenizer.eos_token

print("모델 로드 중...")
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,   # RTX 5090
    device_map="auto",
    attn_implementation="sdpa",
)
model.config.use_cache = False

# ── LoRA 설정 ─────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 데이터셋 ──────────────────────────────────────────────
dataset = VFCoTDataset(DATA_PATH, processor)

# Train / Val 분리 (90/10)
n_val   = max(1, int(len(dataset) * 0.1))
n_train = len(dataset) - n_val
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)
print(f"Train: {n_train}개 / Val: {n_val}개")

# ── 학습 설정 ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,                   # RTX 5090
    tf32=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_dir=LOGGING_DIR,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

# ── Trainer ───────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

# ── 학습 시작 ─────────────────────────────────────────────
print("\n=== 파인튜닝 시작 ===")
print(f"데이터: {n_train}개")
print(f"Epoch: {NUM_EPOCHS}")
print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"LoRA rank: {LORA_RANK}")
print()

trainer.train()

# ── 저장 ──────────────────────────────────────────────────
model.save_pretrained(f"{OUTPUT_DIR}/final")
processor.save_pretrained(f"{OUTPUT_DIR}/final")
print(f"\n모델 저장 완료 → {OUTPUT_DIR}/final")