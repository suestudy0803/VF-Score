# VF-Score: Visual Faithfulness Score

VLM(Vision-Language Model)의 Chain-of-Thought(CoT) 추론이 이미지를 실제로 보고 있는지 정량적으로 측정하는 평가 지표이다.

---

## 배경

VLM은 이미지를 보고 추론하는 것처럼 보이지만, 실제로는 이미지와 무관하게 언어 패턴에만 의존하는 경우가 많다. VF-Score는 이 문제를 측정하기 위해 **counterfactual 이미지**를 활용한다. 진짜로 이미지를 보는 모델이라면 이미지가 바뀔 때 추론도 달라져야 한다는 원리를 이용한다.

---

## 평가 방법

### Counterfactual 유형

| 유형 | 설명 |
|---|---|
| `semantic_swap` | 주 객체를 다른 범주의 객체로 교체 |
| `attribute_flip` | 주 객체의 색상/속성만 변경 |
| `random` | 완전히 무관한 이미지로 교체 |
| `masked` | 검은 화면 |

### VF-Score 수식

```
VF-Score = α × (1 - sim_norm) + (1 - α) × faith_norm
```

- `sim_norm`: 원본 CoT와 변형 CoT의 유사도 (GPT-4o 평가, 1~5점 → 정규화)
- `faith_norm`: 변형 이미지에 대한 CoT의 충실도 (GPT-4o 평가, 1~5점 → 정규화)
- `α = 0.05` (인간 어노테이션과의 Spearman 상관계수 최대화로 최적화)

점수가 높을수록 이미지에 충실한 추론이다.

---

## 실험 결과

### VF-Score (LLaVA-1.6-7B, 200개 샘플)

| CF 유형 | N | 평균 | 표준편차 |
|---|---|---|---|
| semantic_swap | 200 | 0.382 | 0.161 |
| attribute_flip | 200 | 0.224 | 0.126 |
| random | 200 | 0.460 | 0.175 |
| masked | 200 | 0.336 | 0.180 |

### 인간 어노테이션과의 상관관계

| 지표 | 값 | p-value |
|---|---|---|
| Spearman r | 0.301 | 0.0024 |
| Pearson r | 0.283 | 0.0043 |

### 파인튜닝 효과 (HallusionBench)

VF-Score가 높은 샘플로 LLaVA-1.6을 LoRA 파인튜닝한 결과이다.

| 지표 | Baseline | Finetuned | 개선 |
|---|---|---|---|
| aAcc (문항별) | 59.1% | 59.3% | +0.2%p |
| fAcc (이미지별) | 11.3% | 15.1% | **+3.8%p** |

---

## 파이프라인

```
원본 이미지 (ScienceQA subset, 500개)
    ↓
Counterfactual 이미지 생성 (DALL-E / inpainting)
    ↓
LLaVA-1.6-7B CoT 추론 (원본 + 변형 각각)
    ↓
GPT-4o 기반 VF-Score 계산 (similarity + faithfulness)
    ↓
인간 어노테이션 검증 (25개 샘플 × 4 CF 유형)
    ↓
High VF-Score 샘플로 LoRA 파인튜닝
    ↓
HallusionBench / ScienceQA 평가
```

---

## 설치

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
```

Python 3.10+, CUDA GPU 권장 (RTX 3090 이상)

---

## 실행 순서

```bash
# 1. Counterfactual 이미지 생성
python scripts/make_cf_openai.py

# 2. CoT 추론
python scripts/run_cot_inference.py       # 원본 이미지
python scripts/run_cf_cot_inference.py    # 변형 이미지

# 3. GPT-4o 기반 VF-Score 계산
python scripts/compute_final_vf_score.py

# 4. 인간 어노테이션과 상관관계 검증
python scripts/validate_vf_score_v2.py

# 5. 파인튜닝 데이터 준비 및 학습
python scripts/prepare_finetune_data.py
python scripts/finetune_llava.py

# 6. 평가
python scripts/eval_hallusionbench.py
python scripts/eval_scienceqa.py
```

---

## 프로젝트 구조

```
vf-cot/
├── src/
│   ├── metrics/vf_score.py          # VF-Score 핵심 계산
│   └── utils/prompts.py             # CoT 프롬프트 템플릿
├── scripts/
│   ├── make_cf_openai.py            # Counterfactual 이미지 생성
│   ├── run_cot_inference.py         # LLaVA CoT 추론
│   ├── compute_final_vf_score.py    # VF-Score 계산
│   ├── validate_vf_score_v2.py      # 인간 어노테이션 검증
│   ├── finetune_llava.py            # LoRA 파인튜닝
│   ├── eval_hallusionbench.py       # HallusionBench 평가
│   └── analyze_components.py       # 결과 분석 및 시각화
├── results/main_exp/                # 실험 결과 JSON
├── HallusionBench/                  # 서브모듈
└── requirements.txt
```

> 모델 가중치(`models/`)와 원본 이미지(`data/raw/`)는 용량 문제로 포함되지 않는다.
