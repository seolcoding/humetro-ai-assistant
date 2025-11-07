# 100-Question Generation Benchmark System

## 개요

이 실험 시스템은 RAG (Retrieval-Augmented Generation) 시스템의 생성(Generation) 품질을 평가하기 위한 벤치마크 프레임워크입니다. 100개의 테스트 질문을 생성하고, Single-hop과 Multi-hop으로 분류한 후, 여러 LLM 모델의 생성 성능을 비교 평가합니다.

### 주요 특징
- **100개 질문 자동 생성**: RAGAS 프레임워크 활용
- **복잡도 기반 분류**: Single-hop (단순) vs Multi-hop (복합) 질문 분류
- **Generation 중심 평가**: Retrieval 변동성 제거, 순수 생성 능력만 평가
- **재사용 가능한 질문 뱅크**: KG-RAG 실험을 위해 설계
- **다중 모델 지원**: GPT-4o-mini, Ollama (EXAONE, Qwen, Gemma 등)

## 시스템 구조

```
src/rag_pipeline/
├── generate_100q_benchmark.py    # 메인 실험 스크립트
├── question_classifier.py        # Single/Multi-hop 분류 모듈
├── generation_benchmark.py       # Generation 평가 모듈
└── testset_generator.py          # RAGAS 테스트셋 생성 (기존)

data/evaluation/generation_benchmark/
├── raw_testset_*.json           # 원본 100개 질문
├── classification_*.json        # 분류 결과
├── balanced_*.json              # 균형 조정된 50개 질문 (25+25)
├── question_bank_latest.json    # 재사용용 질문 뱅크
├── benchmark_results_*.json     # 모델 평가 결과
└── summary_report_*.md          # 요약 리포트
```

## 설치 및 환경 설정

### 필수 요구사항
```bash
# Python 3.12+ with uv
uv add langchain langchain-openai ragas litellm pandas tqdm
```

### 환경 변수
```bash
export OPENAI_API_KEY="your-api-key"
```

## 사용 방법

### 1. 기본 실행 (100개 질문, 각 25개씩 평가)
```bash
uv run python src/rag_pipeline/generate_100q_benchmark.py
```

### 2. 커스텀 설정
```bash
# 50개 질문 생성, 각 그룹 10개씩 평가
uv run python src/rag_pipeline/generate_100q_benchmark.py \
  --target-size 50 \
  --questions-per-group 10

# 강제 재생성
uv run python src/rag_pipeline/generate_100q_benchmark.py \
  --force-generate

# 분류만 수행 (벤치마크 건너뜀)
uv run python src/rag_pipeline/generate_100q_benchmark.py \
  --skip-benchmark
```

### 3. 옵션 설명
```
--force-generate        # 캐시 무시하고 새로 생성
--num-documents 100     # 소스 문서 수 (기본: 100)
--target-size 100       # 목표 질문 수 (기본: 100)
--questions-per-group 25 # 그룹당 질문 수 (기본: 25)
--classification-method # 분류 방법: llm/rules/hybrid (기본: hybrid)
--skip-benchmark        # 벤치마크 건너뜀 (질문 생성/분류만)
--models               # 평가할 모델 목록
```

## 질문 분류 시스템

### Single-hop vs Multi-hop 기준

**Single-hop 질문**
- 하나의 정보 조각으로 답변 가능
- 단순 검색이나 직접적인 사실 확인
- 예: "2호선 첫차 시간은?", "강남역 주소는?"

**Multi-hop 질문**
- 여러 정보를 조합하거나 추론 필요
- 조건절, 비교, 대안 경로 등 포함
- 예: "2호선이 중단되면 대체 경로는?", "평일과 주말 소요시간 차이는?"

### 분류 방법

1. **Hybrid (기본)**: 규칙 기반 + LLM 조합
   - 명확한 경우: 규칙 기반 (빠름, 비용 효율적)
   - 애매한 경우: GPT-4o-mini로 정밀 분류
   - 신뢰도 점수 제공

2. **LLM**: GPT-4o-mini만 사용 (정확도 높음, 비용 발생)

3. **Rules**: 패턴 매칭만 사용 (빠름, 무료)

## Generation 벤치마크

### 평가 메트릭 (RAGAS)

- **Faithfulness**: 생성된 답변이 제공된 컨텍스트에 근거하는 정도
- **Answer Relevancy**: 질문에 대한 답변의 관련성
- **Answer Correctness**: Ground truth 대비 정확도 (F1 score)

### 고정 컨텍스트 전략

모든 모델에 동일한 retrieved context를 제공하여:
- Retrieval 변동성 제거
- 순수 generation 능력만 비교
- 공정한 모델 간 비교

## 출력 파일 구조

### 1. question_bank_latest.json
```json
{
  "metadata": {
    "name": "Seoul Traffic QA Benchmark v1.0",
    "total_questions": 50,
    "complexity_distribution": {
      "single_hop": 25,
      "multi_hop": 25
    }
  },
  "questions": {
    "single_hop": [...],
    "multi_hop": [...]
  }
}
```

### 2. benchmark_results_*.json
```json
{
  "metadata": {...},
  "single_hop": {
    "GPT-4o-mini": {
      "summary": {
        "faithfulness": 0.85,
        "answer_relevancy": 0.92,
        "answer_correctness": 0.78
      }
    }
  },
  "comparison": {
    "by_complexity": {...},
    "by_model": {...}
  }
}
```

## 재사용성 (KG-RAG 실험용)

생성된 질문 뱅크는 향후 Knowledge Graph 기반 RAG 시스템과 비교 평가에 사용:

1. **현재 (Baseline)**: 전통적 RAG 시스템 평가
2. **향후 (KG-RAG)**: 동일 질문으로 KG-enhanced RAG 평가
3. **비교 분석**: Single-hop vs Multi-hop별 개선도 측정

## 실행 예시

### 전체 파이프라인 실행
```bash
# 100개 질문 생성 → 분류 → 50개 선별 → 벤치마크
uv run python src/rag_pipeline/generate_100q_benchmark.py

# 출력 예시:
======================================================================
                  100-Question Generation Benchmark
======================================================================

[Step 1/5] 질문 생성
✅ 생성된 질문 수: 100

[Step 2/5] 질문 분류 (Single-hop vs Multi-hop)
  - Single-hop: 42 (42.0%)
  - Multi-hop: 58 (58.0%)

[Step 3/5] 균형 조정 (각 25개)
  single_hop: 42 → 25 (신뢰도 기준 선택)
  multi_hop: 58 → 25 (신뢰도 기준 선택)

[Step 4/5] 재사용 가능한 질문 뱅크 저장
💾 질문 뱅크 저장: question_bank_latest.json

[Step 5/5] Generation 벤치마크 실행
평가 중: GPT-4o-mini
평가 중: EXAONE-3.5-7.8B
✅ 벤치마크 완료!
```

### 결과 확인
```bash
# 요약 리포트 보기
cat data/evaluation/generation_benchmark/summary_report_*.md

# 질문 뱅크 확인
cat data/evaluation/generation_benchmark/question_bank_latest.json | jq '.metadata'
```

## 문제 해결

### GPT-5 Temperature 에러
GPT-5는 temperature=1만 지원. RAGAS가 내부적으로 1e-8을 사용하면 에러 발생.
- 해결: judge_model을 "gpt-4o"로 설정 (기본값)

### Ollama 연결 실패
```bash
# Ollama 서버 확인
curl http://100.95.220.92:11434/api/tags

# 모델 설치
ollama pull exaone3.5:7.8b
```

### 메모리 부족
큰 테스트셋 생성 시 메모리 부족 가능:
- target-size를 줄여서 실행
- 단계별 실행 (--skip-benchmark로 생성만)

## 주의사항

1. **API 비용**: 100개 질문 생성 + 분류 + 평가 시 약 $1-2 예상
2. **실행 시간**: 전체 파이프라인 약 10-15분
3. **캐싱**: 동일 설정은 자동 캐싱 (SHA256 기반)
4. **재현성**: 모든 결과는 timestamp와 함께 저장

## 향후 개선 계획

1. **더 정교한 분류**: Reasoning step 수 기반 세분화
2. **추가 메트릭**: Hallucination detection, Coherence
3. **시각화**: 결과 대시보드 자동 생성
4. **CI/CD 통합**: GitHub Actions 자동 벤치마크

## 라이선스

MIT License

## 기여

Issues 및 Pull Requests 환영합니다.

## 문의

프로젝트 관련 문의: [프로젝트 GitHub Issues]