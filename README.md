# VF-Score: Visual Faithfulness Score

VLM(Vision-Language Model)의 Chain-of-Thought 추론이 이미지를 실제로 보고 있는지 측정하는 평가 지표입니다.

## 핵심 아이디어

이미지를 변형(counterfactual)했을 때 모델의 추론이 얼마나 달라지는지를 측정합니다.
진짜로 이미지를 보는 모델이라면 이미지가 바뀌면 추론도 바뀌어야 합니다.

**수식:**
```
VF-Score = α × (1 - sim_norm) + (1 - α) × faith_norm   (α = 0.05)
```
- `sim_norm`: 원본 CoT와 변형 CoT의 유사도 (GPT-4o 평가, 낮을수록 잘 바뀜)
- `faith_norm`: 변형 이미지에 대한 CoT 충실도 (GPT-4o 평가, 높을수록 정확)

## 파이프라인

```
원본 이미지 (ScienceQA)
    ↓
Counterfactual 이미지 생성 (DALL-E)
    ├── semantic_swap  : 주 객체 교체
    ├── attribute_flip : 색상/속성 변경
    ├── random         : 무관한 이미지
    └── masked         : 검은 화면
    ↓
LLaVA-1.6-7B CoT 추론 (원본 + 변형)
    ↓
GPT-4o 기반 VF-Score 계산
    ↓
인간 어노테이션과 상관관계 검증
```

## 실험 결과

| CF 유형 | N | 평균 VF-Score |
|---|---|---|
| semantic_swap | 200 | 0.382 |
| attribute_flip | 200 | 0.224 |
| random | 200 | 0.460 |
| masked | 200 | 0.336 |

점수가 높을수록 이미지에 충실한 추론입니다.

## 설치

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
```

## 실행 순서

```bash
# 1. Counterfactual 이미지 생성
python scripts/make_cf_openai.py

# 2. CoT 추론 (원본 + 변형)
python scripts/run_cot_inference.py
python scripts/run_cf_cot_inference.py

# 3. VF-Score 계산
python scripts/compute_final_vf_score.py

# 4. 결과 분석
python scripts/analyze_components.py
```

## 프로젝트 구조

```
vf-cot/
├── src/metrics/vf_score.py          # VF-Score 계산 핵심 코드
├── src/utils/prompts.py             # CoT 프롬프트
├── scripts/                         # 전체 파이프라인 스크립트
├── results/main_exp/                # 실험 결과
├── HallusionBench/                  # 평가 벤치마크 (submodule)
└── requirements.txt
```

> 모델 가중치(`models/`)와 원본 이미지(`data/raw/`)는 용량 문제로 포함되지 않습니다.
