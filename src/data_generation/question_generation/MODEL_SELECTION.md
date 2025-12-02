# GPT-5.1 Model Selection Rationale

Golden Dataset 질문 생성을 위한 LLM 모델 선택 근거

---

## 1. 모델 선택: GPT-5.1

### 선택 모델
- **Model**: `gpt-5.1`
- **Reasoning Effort**: `medium`
- **Verbosity**: `medium`

### 선택 근거

#### 1.1 최신 Flagship 모델 (2025-11-12 출시)

GPT-5.1은 OpenAI의 최신 flagship 모델로, 이전 GPT-5 대비 주요 개선점:

| 항목 | GPT-5 | GPT-5.1 |
|------|-------|---------|
| SWE-bench Verified | 72.8% | **76.3%** |
| 토큰 효율성 | baseline | **-23%** |
| 복잡 작업 정확도 | baseline | **+18%** |
| 처리 속도 | baseline | **2-3x faster** |

**출처**: [OpenAI GPT-5.1 for Developers](https://openai.com/index/gpt-5-1-for-developers/)

#### 1.2 Reasoning 능력

GPT-5.1은 Artificial Analysis Intelligence Index에서:
- High: 68점 (최고)
- Medium: 67점 (o3 수준)
- Low: 64점 (DeepSeek R1 ~ o3 사이)

**질문 생성 작업에 Medium 선택 이유**:
- 질문 생성은 "적절한 복잡도"의 작업
- High: 과도한 추론 → 비용/시간 증가
- Low/None: 단순 추론 → 다양한 질문 유형 생성 어려움
- **Medium: 균형점** - Multi-hop 추론 경로 설계에 적합

**출처**: [Artificial Analysis GPT-5 Benchmarks](https://artificialanalysis.ai/articles/gpt-5-benchmarks-and-analysis)

#### 1.3 한국어 행정문서 처리

GPT-5.1 선택 이유 (vs 한국어 특화 모델):

| 고려 모델 | 장점 | 단점 | 판단 |
|----------|------|------|------|
| GPT-5.1 | 최고 추론력, 지시 따르기 | 한국어 최적화 X | **선택** |
| Claude 4.5 | 긴 문맥 처리 | 느린 속도 | X |
| EXAONE-3.5 | 한국어 특화 | 추론력 제한 | X |
| Gemini 2.0 | 멀티모달 | JSON 출력 불안정 | X |

**판단 근거**:
1. 질문 생성은 **추론 능력**이 핵심 (한국어 유창성보다 중요)
2. Multi-hop 질문은 논리적 추론 경로 설계 필요
3. GPT-5.1의 지시 따르기 능력이 JSON 출력에 유리
4. 400K 컨텍스트 윈도우로 관련 문서 동시 처리 가능

#### 1.4 JSON Structured Output 지원

GPT-5.1의 출력 제어 기능:

```python
# Context-Free Grammar (CFG) 지원
# → JSON 스키마 강제 가능

response = client.responses.create(
    model="gpt-5.1",
    input=prompt,
    reasoning={"effort": "medium"},
    text={"verbosity": "medium"},
    # JSON 출력 보장
)
```

**장점**:
- 일관된 JSON 포맷 출력
- 파싱 실패 최소화
- 질문 유형별 필드 검증 용이

---

## 2. Reasoning Effort 선택: Medium

### 2.1 Effort 레벨별 특성

| Effort | 토큰 사용 | 속도 | 적합 작업 |
|--------|----------|------|----------|
| **none** | 최소 | 최고 | 단순 분류, 추출 |
| **low** | 소 | 빠름 | 간단한 QA |
| **medium** | 중 | 보통 | **복잡한 생성 작업** |
| **high** | 대 | 느림 | 수학, 코딩 |

**출처**: [GPT-5 Reasoning Effort Levels](https://www.arsturn.com/blog/gpt-5-reasoning-effort-levels-explained)

### 2.2 Medium 선택 이유

**질문 생성 작업 특성**:

```
1. 문서 이해 → 핵심 정보 추출
2. 질문 유형 결정 → 적절한 난이도
3. Multi-hop 경로 설계 → 논리적 연결
4. 답변 생성 → 문서 기반 정확성
```

- **Low**: 단순 Factoid만 생성 가능, Multi-hop 경로 설계 부족
- **Medium**: 2-3 hop 추론 경로 설계에 적합
- **High**: 질문 생성에 과도한 리소스 (수학 문제 수준 불필요)

### 2.3 비용-성능 트레이드오프

| Effort | 상대 비용 | 품질 (예상) |
|--------|----------|------------|
| low | 1x | 60% |
| **medium** | 2-3x | **85%** |
| high | 5-8x | 95% |

**Medium이 질문 생성에 최적**:
- 180개 질문 생성 시 비용 합리적
- Multi-hop 품질 충분
- Human-in-the-Loop로 최종 검증 예정

---

## 3. Verbosity 선택: Medium

### 3.1 Verbosity 레벨별 특성

| Verbosity | 출력 길이 | 특성 |
|-----------|----------|------|
| low | 짧음 | 간결한 답변, 설명 최소 |
| **medium** | 보통 | 균형잡힌 설명 |
| high | 김 | 상세한 설명, 코드 주석 |

### 3.2 Medium 선택 이유

**질문 생성 출력 구조**:

```json
{
  "question": "질문 (필수)",
  "answer": "답변 (필수)",
  "reasoning_steps": ["Step 1", "Step 2"],  // Multi-hop
  "evidence": "근거 문장"
}
```

- **Low**: `reasoning_steps` 생략 가능성
- **Medium**: 필수 필드 + 추론 단계 포함
- **High**: 과도한 설명으로 JSON 파싱 복잡

---

## 4. 대안 모델 검토

### 4.1 검토 모델

| 모델 | Intelligence Index | 한국어 | JSON 출력 | 판단 |
|------|-------------------|--------|----------|------|
| GPT-5.1 | 68 (high) | O | 우수 | **선택** |
| Claude 4.5 | 65 | O | 좋음 | 차선 |
| Gemini 2.0 | 62 | O | 불안정 | X |
| DeepSeek R1 | 64 | X | 보통 | X |

### 4.2 GPT-5.1 vs Claude 4.5

| 항목 | GPT-5.1 | Claude 4.5 |
|------|---------|------------|
| 추론력 | **우수** | 좋음 |
| 지시 따르기 | **매우 우수** | 우수 |
| JSON 출력 | **매우 안정** | 안정 |
| 비용 | 높음 | 높음 |
| 속도 | **빠름** | 보통 |

**결론**: GPT-5.1이 질문 생성에 더 적합

---

## 5. 구현 파라미터

```python
# 최종 선택 파라미터
MODEL_CONFIG = {
    "model": "gpt-5.1",
    "reasoning": {
        "effort": "medium"  # 복잡한 생성 작업에 적합
    },
    "text": {
        "verbosity": "medium"  # 필수 필드 + 추론 단계
    },
    "max_output_tokens": 2000  # 충분한 출력 공간
}
```

### 5.1 질문 유형별 조정 (선택적)

| 질문 유형 | Effort | Verbosity | 이유 |
|----------|--------|-----------|------|
| Simple Factoid | low | low | 단순 추출 |
| Constraint | medium | medium | 조건 파악 필요 |
| Multi-hop 2 | **medium** | **medium** | 추론 경로 필요 |
| Multi-hop 3 | **medium** | **medium** | 복잡한 체인 |
| Reasoning | medium | medium | 인과관계 파악 |

→ **일관성을 위해 모든 유형에 medium/medium 적용**

---

## 6. 참고문헌

1. OpenAI. (2025). [GPT-5.1 for Developers](https://openai.com/index/gpt-5-1-for-developers/)
2. OpenAI. (2025). [GPT-5.1 Prompting Guide](https://cookbook.openai.com/examples/gpt-5/gpt-5-1_prompting_guide)
3. Artificial Analysis. (2025). [GPT-5 Benchmarks and Analysis](https://artificialanalysis.ai/articles/gpt-5-benchmarks-and-analysis)
4. Arsturn. (2025). [GPT-5 Reasoning Effort Levels Explained](https://www.arsturn.com/blog/gpt-5-reasoning-effort-levels-explained)

---

**작성일**: 2025-12-02
**버전**: 1.0
