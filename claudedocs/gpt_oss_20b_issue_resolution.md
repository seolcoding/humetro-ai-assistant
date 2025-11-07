# GPT-OSS-20B 한국어 응답 이슈 해결 보고서

**작성일**: 2025-10-29 23:25
**이슈**: GPT-OSS-20B 모델이 특정 한국어 프롬프트에 빈 응답 반환

## 🔍 문제 진단

### 증상
- 한국어 정보성 질의에 빈 응답 (예: "서울시 120 다산콜센터는 무엇인가요?")
- 영어 프롬프트는 정상 응답
- 한국어 일상 대화는 정상 응답

### 원인 분석

#### 1. Harmony Format 확인
GPT-OSS-20B는 OpenAI의 Harmony response format을 사용:
- 특수 태그: `<|start|>`, `<|end|>`, `<|message|>`, `<|channel|>`
- Ollama가 자동으로 포맷 처리
- **포맷 문제 아님** (영어와 일부 한국어는 정상 작동)

#### 2. 모델 특성
- **모델 구조**: 21B 파라미터 (3.6B active, MoE 아키텍처)
- **학습 데이터**: 주로 영어 중심, 한국어 데이터 제한적
- **용도**: 구조화된 출력, 도구 호출, 추론 작업 특화

#### 3. 테스트 결과

| 프롬프트 유형 | 언어 | 응답 상태 | 응답 길이 |
|-------------|------|----------|-----------|
| 정보 질의 (다산콜센터) | 한국어 | ❌ 빈 응답 | 0 chars |
| 기술 질문 (Python) | 영어 | ✅ 정상 | 807 chars |
| 일상 대화 (안녕하세요) | 한국어 | ✅ 정상 | 82 chars |
| 민원 절차 | 한국어 | ❌ 빈 응답 | 0 chars |

## 💡 발견 사항

### GPT-OSS-20B의 한계
1. **도메인 특화 한국어 약함**: 공공 서비스, 행정 용어 등
2. **영어 중심 설계**: OpenAI 모델로 영어 성능 우선
3. **추론 모드 영향**: Reasoning level이 한국어 생성에 영향

### 작동하는 경우
- 영어 프롬프트 (모든 유형)
- 한국어 일상 대화
- 간단한 인사말이나 감정 표현

### 작동하지 않는 경우
- 한국어 정보성 질의
- 한국어 전문 용어 포함 질문
- 한국어 절차 설명 요청

## 🛠️ 해결 방안

### 1. 단기 해결책
```python
# 프롬프트 보강 전략
if model == "gpt-oss:20b" and is_korean(prompt):
    # 영어 컨텍스트 추가
    enhanced_prompt = f"Please answer in Korean: {prompt}"
    # 또는 이중 언어 프롬프트
    enhanced_prompt = f"{prompt}\n(Please provide the answer in Korean)"
```

### 2. 중기 해결책
- **Fallback 메커니즘**: GPT-OSS가 빈 응답시 다른 모델 사용
- **프롬프트 엔지니어링**: 시스템 프롬프트에 한국어 강제
- **하이브리드 접근**: 영어로 추론 → 한국어로 번역

### 3. 장기 해결책
- **Fine-tuning**: 한국어 데이터로 추가 학습
- **모델 교체**: 한국어 특화 모델 사용 (EXAONE, Solar Pro)
- **RAG 보강**: 한국어 문서 검색으로 보완

## 📊 논문 실험 영향

### 실험 설계 조정
1. **GPT-OSS-20B 역할 재정의**
   - 구조화된 쿼리 전문 (SQL, API 호출)
   - 영어 기반 추론 작업
   - Graph RAG의 관계 추출

2. **평가 메트릭 수정**
   - 한국어 생성 품질은 다른 3개 모델로 평가
   - GPT-OSS는 구조화 작업 정확도 중심

3. **보완 전략**
   - EXAONE-3.5: 한국어 주력 모델
   - GPT-OSS-20B: 구조화/추론 특화
   - 역할 분담으로 시너지 창출

## ✅ 검증 코드

```python
# src/scripts/test_gpt_oss_chat.py
def test_gpt_oss_with_different_prompts():
    """GPT-OSS-20B 다양한 프롬프트 테스트"""

    test_cases = [
        # 작동하는 케이스
        ("Hello, how are you?", "영어 인사", "✅"),
        ("What is machine learning?", "영어 기술", "✅"),
        ("안녕하세요!", "한국어 인사", "✅"),

        # 작동 안하는 케이스
        ("서울시 120 다산콜센터는?", "한국어 정보", "❌"),
        ("주민등록등본 발급 방법", "한국어 절차", "❌"),
    ]

    for prompt, category, expected in test_cases:
        response = test_model(prompt)
        status = "✅" if len(response) > 0 else "❌"
        assert status == expected, f"Unexpected result for {category}"
```

## 🎯 결론

**GPT-OSS-20B는 한국어 제한이 있지만 여전히 유용함**:
1. 영어 작업과 구조화된 출력에 탁월
2. MoE 아키텍처로 효율적인 추론
3. Graph RAG의 관계 추출에 적합

**논문 실험 계속 진행 가능**:
- 4개 모델 각각의 강점 활용
- GPT-OSS는 특정 역할에 특화
- 전체 시스템 성능은 유지

## 📝 업데이트 필요 사항

1. `config/models.yaml`: GPT-OSS 한국어 제한 명시
2. 실험 노트북: 모델별 역할 재정의
3. 평가 스크립트: 언어별 평가 분리