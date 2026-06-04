# VF-Score: Visual Faithfulness Score for VLM Chain-of-Thought Reasoning

A metric for evaluating how visually faithful the Chain-of-Thought (CoT) reasoning of Vision-Language Models (VLMs) is to the actual image content.

## Overview

VF-Score measures whether a VLM truly "looks" at an image when reasoning, rather than relying on language priors or hallucination. The core idea: if a model's reasoning is genuinely grounded in the image, its CoT should change meaningfully when the image is modified.

**Formula:**

```
VF-Score = α × (1 - sim_norm) + (1 - α) × faith_norm
```

- `sim_norm`: GPT-4o-judged similarity between original and counterfactual CoT (normalized 0–1)
- `faith_norm`: GPT-4o-judged faithfulness of counterfactual CoT to the modified image (normalized 0–1)
- `α = 0.05` (optimized against human annotations)

## Pipeline

```
Raw Images (ScienceQA)
        │
        ▼
Counterfactual Generation (DALL-E / inpainting / random)
        │
        ├── semantic_swap   (replace main object)
        ├── attribute_flip  (change color/attribute)
        ├── random          (unrelated image)
        └── masked          (black image)
        │
        ▼
CoT Inference (LLaVA-1.6-7B)
  ├── CoT for original image
  └── CoT for counterfactual image
        │
        ▼
VF-Score Computation (GPT-4o as judge)
  ├── Similarity score  (1–5): how much the CoT changed
  └── Faithfulness score (1–5): how accurately CoT describes the new image
        │
        ▼
Validation (Spearman correlation with human annotations)
```

## Results

Mean VF-Scores across 200 samples (4 counterfactual types):

| CF Type        |  N  | Mean  |  Std  |  Min  |  Max  |
|----------------|-----|-------|-------|-------|-------|
| semantic_swap  | 200 | 0.382 | 0.161 | 0.099 | 0.856 |
| attribute_flip | 200 | 0.224 | 0.126 | 0.017 | 0.706 |
| random         | 200 | 0.460 | 0.175 | 0.116 | 0.888 |
| masked         | 200 | 0.336 | 0.180 | 0.017 | 0.931 |

Higher VF-Score = more visually faithful reasoning.

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, CUDA GPU (recommended: RTX 3090+)

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

## Usage

### Step 1: Generate counterfactual images
```bash
python scripts/make_cf_openai.py
```

### Step 2: Run CoT inference (LLaVA-1.6)
```bash
python scripts/run_cot_inference.py          # original images
python scripts/run_cf_cot_inference.py       # counterfactual images
```

### Step 3: Compute VF-Score
```bash
python scripts/compute_final_vf_score.py
```

### Step 4: Analyze results
```bash
python scripts/analyze_components.py
python scripts/visualize_alpha.py
```

## Project Structure

```
vf-cot/
├── src/
│   ├── metrics/
│   │   └── vf_score.py          # Core VF-Score computation
│   └── utils/
│       └── prompts.py           # CoT prompt templates
├── scripts/
│   ├── run_cot_inference.py     # LLaVA CoT generation
│   ├── run_cf_cot_inference.py  # Counterfactual CoT generation
│   ├── make_cf_openai.py        # Counterfactual image generation
│   ├── compute_final_vf_score.py # GPT-4o based VF-Score
│   ├── compute_ai_scores.py     # AI judge scoring
│   └── analyze_components.py   # Result analysis
├── data/
│   ├── raw/                     # Source images & QA pairs
│   ├── counterfactual/          # Generated counterfactual images
│   ├── cot_outputs/             # LLaVA CoT outputs
│   └── annotations/             # Human annotation results
├── results/
│   └── main_exp/                # VF-Score results & summaries
├── HallusionBench/              # HallusionBench evaluation
├── SPD-Faith-Bench/             # SPD-Faith-Bench evaluation
└── requirements.txt
```

## Related Benchmarks

- [HallusionBench](https://github.com/tianyi-lab/HallusionBench) — visual hallucination benchmark
- [SPD-Faith-Bench](SPD-Faith-Bench/) — spatial/perceptual faithfulness benchmark

## Notes

- Model weights (`models/`) and raw data (`data/raw/`, `data/counterfactual/`) are not included in this repository due to size.
- All API keys must be set as environment variables — never hardcoded.
