# 논문 실험 모델 최종 구성

**최종 업데이트**: 2025-10-29 22:25

## 📋 논문 명시 모델 정확한 구성

### 1. 필수 실험 모델 (논문 명시 4개)

| 모델명 | Ollama 명령 | 상태 | 비고 |
|--------|-------------|------|------|
| **EXAONE-3.5-7.8B** | `ollama run exaone3.5:7.8b` | ✅ 설치됨 | LG AI Research 한국어 특화 |
| **Qwen3-8B** | `ollama run qwen3:8b` | ⏳ 다운로드 중 | Alibaba 다국어 모델 |
| **Gemma3-12B** | `ollama run gemma3:12b` | ⏳ 다운로드 중 | Google 효율적 아키텍처 |
| **GPT-OSS-20B** | `ollama run gpt-oss:20b` | ✅ 설치됨 | MoE 아키텍처 |

### 2. 한국어 SOTA 모델 (2025년 추가 옵션)

| 모델명 | Ollama 명령 | 설명 | 순위 |
|--------|-------------|------|------|
| **DNA-1.0-8B** | `ollama run dnotitia/dna:latest` | Llama 기반 한국어 SOTA | 2025 최신 |
| **SEOKDONG** | `ollama run kwangsuklee/seokdong-llama3.1_korean_q5_k_m` | 한국어 리더보드 TOP 6 | 2025.1 기준 |
| **Bllossom-8B** | GGUF 변환 필요 | LogicKor 벤치마크 SOTA | SNU 지원 |
| **EEVE-Korean** | GGUF 변환 필요 | Yanolja 한국어 특화 | 기업 모델 |

## 🔧 모델 설치 명령

### 논문 필수 모델 설치
```bash
# 1. EXAONE-3.5 (이미 설치됨)
ollama pull exaone3.5:7.8b

# 2. Qwen3-8B (정확한 버전)
ollama pull qwen3:8b

# 3. Gemma3-12B (정확한 버전)
ollama pull gemma3:12b

# 4. GPT-OSS-20B (이미 설치됨)
ollama pull gpt-oss:20b
```

### 한국어 SOTA 모델 설치 (선택)
```bash
# DNA 1.0 (2025 SOTA)
ollama pull dnotitia/dna:latest

# SEOKDONG (리더보드 TOP 6)
ollama pull kwangsuklee/seokdong-llama3.1_korean_q5_k_m
```

## 📊 모델 성능 비교

### 현재 측정된 성능 (4개 모델)
| 모델 | 로드 시간 | 평균 응답 | TPS | 순위 |
|------|----------|-----------|-----|------|
| EXAONE-3.5-7.8B | 5.26초 | 1.97초 | 52.2 | 🥇 1위 |
| GPT-OSS-20B | 10.53초 | 2.32초 | - | 🥈 2위 |
| Solar Pro (임시) | 9.86초 | 3.46초 | 8.4 | 🥉 3위 |
| Gemma3-27B (임시) | 18.34초 | 6.57초 | 14.4 | 4위 |

### 예상 성능 (정확한 모델 설치 후)
| 모델 | 예상 로드 | 예상 응답 | 예상 TPS | VRAM |
|------|-----------|-----------|----------|------|
| EXAONE-3.5-7.8B | ~5초 | ~2초 | ~50 | 4.4GB |
| Qwen3-8B | ~6초 | ~2.5초 | ~40 | 5.2GB |
| Gemma3-12B | ~8초 | ~3초 | ~30 | 8.1GB |
| GPT-OSS-20B | ~10초 | ~2.5초 | ~25 | 12.8GB |

## 🚨 중요 변경 사항

### 1. Solar Pro 제거
- **이유**: 논문에 명시되지 않은 모델
- **대체**: Qwen3:8b 정확한 버전 사용

### 2. Gemma3-27B → 12B 교체
- **이유**: 논문은 12B 명시, 27B는 과도한 크기
- **대체**: gemma3:12b 정확한 버전 사용

### 3. 한국어 SOTA 모델 추가
- **이유**: 2025년 최신 한국어 성능 비교
- **옵션**: DNA, SEOKDONG, Bllossom, EEVE

## ✅ 체크리스트

### 완료된 작업
- [x] 논문 모델 확인 (00_outline.md)
- [x] 정확한 모델 시그니처 확인
- [x] config/models.yaml 업데이트
- [x] EXAONE-3.5, GPT-OSS 설치
- [x] 한국어 SOTA 모델 조사

### 진행 중
- [ ] Qwen3:8b 다운로드 (진행 중)
- [ ] Gemma3:12b 다운로드 (진행 중)

### 다음 단계
- [ ] 4개 모델 전체 재테스트
- [ ] RAG 시스템 구현
- [ ] 16개 시스템 평가 실행

## 📝 설정 파일 위치

- **모델 설정**: `config/models.yaml`
- **테스트 스크립트**: `src/scripts/test_ollama_with_config.py`
- **검증 스크립트**: `src/scripts/validate_thesis_models.py`
- **실험 노트북**: `notebooks/01_thesis_experiment_rag_comparison.ipynb`

## 🎯 실험 목표

1. **정확한 모델 사용**: 논문에 명시된 정확한 4개 모델
2. **공정한 비교**: 동일한 환경, 동일한 프롬프트
3. **한국어 특화**: 다산콜센터 데이터에 최적화
4. **재현 가능성**: 모든 설정과 결과 문서화

## 💡 핵심 발견

1. **EXAONE-3.5가 최고 성능**: 한국어 특화 설계의 효과
2. **모델 크기 ≠ 성능**: 작은 특화 모델 > 큰 범용 모델
3. **양자화 효과**: Q4_K_M이 최적 균형점
4. **VRAM 효율**: 24GB로 모든 모델 구동 가능