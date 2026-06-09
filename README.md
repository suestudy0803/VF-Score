# VF-Score: Visual Faithfulness Score for Multimodal Chain-of-Thought

> Visual Language Model(VLM)이 CoT 추론 시 이미지를 실제로 보고 있는지 측정하는 자동 평가 지표

## Overview

VF-Score는 Counterfactual 이미지 교체를 통해 VLM의 Visual Faithfulness를 행동 기반(behavioral)으로 측정하는 프레임워크입니다.

```
VF-Score = α × (1 - sim_norm) + (1 - α) × faith_norm
```

- `sim_norm`: 원본 CoT와 변형 CoT의 의미적 유사도 (GPT-4o 평가, 1~5점 → 정규화)
- `faith_norm`: 변형 이미지에 대한 CoT의 faithfulness (GPT-4o + 이미지 평가, 1~5점 → 정규화)
- `α`: 최적 가중치 (실험으로 결정, α=0.05)

## Counterfactual Image Types

| 유형 | 설명 | 생성 방법 |
|------|------|----------|
| Semantic Swap | 핵심 물체를 다른 카테고리로 교체 | gpt-image-1 API |
| Attribute Flip | 색깔 등 속성만 변경 | gpt-image-1 API |
| Random | 무관한 이미지로 교체 | COCO 랜덤 샘플링 |
| Masked | 검은 화면으로 대체 | PIL Image.new() |

## Key Results

- Human Annotation과의 상관관계: Spearman **ρ = 0.66** (이미지 포함 평가)
- VF-Score 기반 데이터 필터링 후 LLaVA-1.6 파인튜닝
- HallusionBench fAcc: **+1.89%** 향상

## Environment

```bash
GPU: RTX 5090
CUDA: 12.8
Python: 3.11
PyTorch: cu128
```

## Installation

```bash
git clone https://github.com/suestudy0803/VF-Score.git
cd VF-Score
pip install -r requirements.txt
```

## Project Structure

```
vf-cot/
└── scripts/
    ├── data/
    │   ├── raw/                        # ScienceQA 서브셋
    │   ├── counterfactual/             # CF 이미지 경로
    │   ├── cot_outputs/                # LLaVA CoT 결과
    │   └── annotations/                # Human Annotation
    ├── results/
    │   └── main_exp/                   # VF-Score 계산 결과
    ├── compute_final_vf_scores_v2.py   # VF-Score 계산
    ├── eval_hallusionbench.py          # HallusionBench 평가
    └── finetune_llava.py               # LLaVA LoRA 파인튜닝
```

## Pipeline

```
1. ScienceQA 500개 샘플링
        ↓
2. LLaVA-1.6으로 원본 CoT 생성
        ↓
3. Counterfactual 이미지 4종 생성
        ↓
4. 변형 이미지로 CoT 재생성
        ↓
5. VF-Score 계산 (GPT-4o, α=0.05)
        ↓
6. High VF-Score 데이터 필터링 (≥0.5, 122개)
        ↓
7. LLaVA-1.6 LoRA 파인튜닝
        ↓
8. HallusionBench 평가
```

## Citation

```bibtex
@misc{lee2025vfscore,
  title  = {Visual Faithfulness in Multimodal Chain-of-Thought},
  author = {Yunseo Lee},
  year   = {2025},
  url    = {https://github.com/suestudy0803/VF-Score}
}
```

## Reference

- Wei et al., Chain-of-Thought Prompting, NeurIPS 2022
- Zhang et al., Multimodal-CoT, ACL 2023
- Jing et al., FaithScore, EMNLP Findings 2024
- Guan et al., HallusionBench, CVPR 2024
- Uppaal et al., Journey Before Destination, arXiv 2025
