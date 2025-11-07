# 골든 데이터셋 (Golden Test Dataset)

## 개요

모든 RAG 벤치마크 평가에서 표준으로 사용할 50개 질문-답변 세트입니다.
일관된 평가를 위해 고정된 테스트셋으로, 서울 교통 관련 실제 사용 시나리오를 반영합니다.

## 데이터셋 정보

### 위치
- **CSV**: `data/evaluation/testsets/golden_testset_50.csv`
- **JSON**: `data/evaluation/testsets/golden_testset_50.json`

### 구성
- **총 질문 수**: 50개
  - Single-hop: 25개 (단순 정보 검색)
  - Multi-hop: 25개 (복합 추론 필요)
- **언어**: 모든 질문과 답변이 한국어로 작성됨
- **도메인**: 서울 교통 (지하철, 버스, 대중교통 정책)

### 통계
- 평균 답변 길이: 229자
- 최단 답변: 52자
- 최장 답변: 413자

## 사용 방법

### 기본 사용 (전체 50개 질문)

```json
{
  "questions": {
    "source": "golden"
  }
}
```

`source`가 지정되지 않으면 자동으로 골든 데이터셋을 사용합니다.

### 일부 질문만 사용 (순차적)

```json
{
  "questions": {
    "source": "golden",
    "limit": 10
  }
}
```

처음 10개 질문만 사용합니다.

### 랜덤 샘플링

```json
{
  "questions": {
    "source": "golden",
    "limit": 15,
    "random_sample": true,
    "random_seed": 42
  }
}
```

골든 데이터셋에서 무작위로 15개 질문을 샘플링합니다.
`random_seed`를 고정하면 매번 같은 질문들이 선택됩니다.

## 골든 데이터셋을 사용하는 Config 파일

다음 config 파일들이 골든 데이터셋을 사용하도록 업데이트되었습니다:

### Healthcheck (빠른 검증)
- **`config/healthcheck_simple.json`**: 2 questions, 2 models, Naive RAG only
- **`config/healthcheck_3way.json`**: 2 questions, 2 models, 3 RAG methods

### Production Benchmark
- **`config/compare_3way_rag.json`**: 50 questions, 6 models, 3 RAG methods

## 이전 테스트셋 Archive

기존 테스트셋들은 `data/evaluation/testsets/archive/`로 이동되었습니다:
- `testset_359312cb043d5a69.*` - 이전 50개 질문 세트
- `testset_ab9011d2e08ae32f.*` - 이전 2개 질문 세트 (healthcheck)
- 기타 실험용 테스트셋들

필요시 archive에서 복구하여 사용할 수 있습니다.

## 데이터 생성 과정

골든 데이터셋은 다음 과정을 통해 생성되었습니다:

1. **초기 생성**: RAGAS를 사용하여 서울 교통 문서에서 질문-답변 생성
2. **균형 조정**: Single-hop 25개 + Multi-hop 25개로 균형 맞춤
3. **번역**: 모든 영어 질문과 답변을 한국어로 번역
4. **품질 개선**: 어색한 표현 개선 ("함의" → "내용/의미", "세부사항" → "구체적인 내용" 등)
5. **검증**: 번역 품질 및 질문-답변 매칭 확인

## 예시 질문

### Single-hop (단순 정보 검색)
```
Q: 여의도역 우회에 대한 중요정보는?
A: 여의도역은 2024년 12월 14일부터 15일까지 집회로 인해 통제되는 구간에 위치하고 있으며,
   5623번, 5615번, 5618번 노선의 우회...
```

### Multi-hop (복합 추론)
```
Q: 서울의 장거리 버스 노선 개선이 승객 안전과 버스 운전기사의 근무 환경을
   어떻게 향상시킬 것으로 예상되나요?
A: 서울의 장거리 버스 노선 개선에는 60km를 초과하는 노선의 대대적인 개편이 포함되며,
   이는 승객 안전을 강화하고 버스 운전기사의 근무 환경을 개선하기 위한 것입니다...
```

## 주의사항

1. **일관성**: 모든 벤치마크 비교는 동일한 질문 세트를 사용해야 합니다
2. **재현성**: `random_seed`를 고정하여 랜덤 샘플링 결과를 재현할 수 있습니다
3. **확장**: 새로운 질문이 필요한 경우, 골든 데이터셋과 별도로 관리하세요
4. **보존**: 골든 데이터셋 파일은 수정하지 않고 그대로 유지하세요

## 업데이트 이력

- **2025-11-06**: 초기 골든 데이터셋 생성 및 설정
  - 50개 질문 (Single-hop 25 + Multi-hop 25)
  - 모든 질문/답변 한국어 번역 완료
  - 주요 config 파일 업데이트
  - Archive 폴더로 이전 테스트셋 이동
