# Humetro AI Assistant - 프로젝트 현황 문서

## 📋 프로젝트 개요

### 기본 정보
- **프로젝트명**: Humetro AI Assistant
- **버전**: 0.1.0
- **목적**: 부산 도시철도(Humetro) 이용자를 위한 AI 기반 대화형 어시스턴트
- **핵심 기술**: RAG (Retrieval-Augmented Generation)
- **주요 언어**: Python 3.12
- **라이선스**: MIT

### 프로젝트 목표
1. 부산 도시철도 이용자에게 정확한 정보 제공
2. 실시간 경로 안내 및 요금 계산
3. 다국어 지원 (한국어, 영어, 중국어, 일본어)
4. 지속적으로 업데이트되는 정보 제공

## 🏗️ 시스템 아키텍처

### 전체 아키텍처
```
┌─────────────────────────────────────────────────────────┐
│                    사용자 인터페이스                      │
│                  (Streamlit Web App)                     │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    LangChain Agent                       │
│              (대화 관리 및 도구 오케스트레이션)          │
└─────────────────────────────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  LLM Tools   │  │  RAG System  │  │  External    │
│  (도구 모음) │  │ (검색 시스템)│  │    APIs      │
└──────────────┘  └──────────────┘  └──────────────┘
```

### 핵심 구성 요소

#### 1. RAG 파이프라인
- **Retrieval**: BM25 + Semantic Search (BAAI/bge-small-en-v1.5)
- **Reranking**: MonoT5 모델
- **Generation**: GPT-4o-mini
- **Vector Store**: FAISS
- **Query Expansion**: GPT-4o-mini 기반

#### 2. 데이터 처리 파이프라인
```
웹 크롤링 → 문서 청킹 → 임베딩 생성 → 벡터 DB 저장 → QA 데이터셋 생성
```

## 📁 프로젝트 구조

```
humetro-ai-assistant/
├── 📂 src/                         # 소스 코드
│   ├── 📂 llm_tools/               # LLM 도구 모음
│   │   ├── HumetroWikiSearch.py    # Wiki 검색 도구
│   │   ├── TMapTool.py             # 경로 안내 도구
│   │   ├── HumetroFare.py          # 요금 계산 도구
│   │   ├── HumetroSchedule.py      # 시간표 조회 도구
│   │   ├── HumetroWebSearch.py     # 웹 검색 도구
│   │   ├── StationDistanceTool.py  # 역간 거리 계산
│   │   ├── StationInformationTool.py # 역 정보 조회
│   │   └── basic_tools.py          # 기본 유틸리티
│   │
│   ├── 📂 data_processor/          # 데이터 처리
│   │   ├── crawler.py              # 웹 크롤러
│   │   ├── generate_qa_data.py     # QA 데이터 생성
│   │   ├── evaluate_rag.py         # RAG 평가
│   │   └── rag_config.yaml         # RAG 설정
│   │
│   ├── 📂 legacy/                  # 레거시 코드
│   │   ├── app.py                  # Streamlit 앱
│   │   ├── haa_agent.py            # Agent 실행기
│   │   └── mongo_db.py             # MongoDB 연동
│   │
│   └── 📂 common/                  # 공통 유틸리티
│       ├── get_root.py             # 프로젝트 루트 경로
│       └── hwp_to_pdf_converter.py # HWP 변환
│
├── 📂 datasets/                    # 데이터셋
│   ├── final_docs/                 # 처리된 문서
│   └── rag_evaluation/             # 평가 결과
│
├── 📂 crawl_result/                # 크롤링 결과
├── 📂 vectorstore/                 # 벡터 저장소
├── 📂 notebooks/                   # 개발용 노트북
│
├── 📄 main.py                      # CLI 진입점
├── 📄 pyproject.toml               # 프로젝트 설정
├── 📄 requirements.txt             # 의존성 패키지
└── 📄 README.md                    # 프로젝트 설명서
```

## 🛠️ 주요 기능

### 1. 정보 검색 및 응답
- **Wiki 검색**: 환승, 정기승차권, 운임반환 등 정책 정보
- **실시간 웹 검색**: 최신 공지사항 및 변경 사항
- **RAG 기반 응답**: 크롤링된 데이터 기반 정확한 답변

### 2. 경로 안내 서비스
- **TMap API 연동**: 실시간 대중교통 경로 안내
- **Kakao API 연동**: 목적지 좌표 검색
- **요금 계산**: 연령별 할인 적용 요금 산출
- **다중 경로 제공**: 지하철 우선/버스 포함 옵션

### 3. 역 정보 서비스
- **역 정보 조회**: 시설, 출구, 편의시설 정보
- **역간 거리 계산**: 두 역 사이 거리 및 소요시간
- **시간표 조회**: 첫차/막차 시간 정보

### 4. 다국어 지원
- 한국어 (기본)
- 영어
- 중국어
- 일본어

## 💻 기술 스택

### 핵심 프레임워크
- **LangChain** (0.3.25): Agent 오케스트레이션
- **OpenAI API** (1.75.0): LLM 기능
- **AutoRAG** (0.3.13): RAG 파이프라인 최적화
- **Streamlit** (1.45.0): 웹 인터페이스

### 벡터 데이터베이스
- **ChromaDB** (1.0.8)
- **Milvus** (2.4.12)
- **Pinecone** (6.0.2)
- **Weaviate** (4.14.3)
- **Qdrant** (1.14.2)

### 임베딩 및 ML
- **Transformers** (4.44.2)
- **Torch** (2.7.0)
- **Sentence-transformers**
- **FAISS**: 벡터 검색

### 데이터 처리
- **Pandas** (2.2.3)
- **NumPy** (1.26.4)
- **BeautifulSoup4** (4.13.4)
- **Crawl4ai** (0.6.2)

### 평가 도구
- **Ragas** (0.2.15): RAG 평가
- **LangSmith** (0.3.42): LLM 모니터링

## 📊 데이터 플로우

### 1. 데이터 수집 단계
```
Humetro 웹사이트 → 크롤러 → Markdown 파일 → 전처리
```

### 2. RAG 준비 단계
```
문서 청킹 (512 tokens) → 임베딩 생성 → FAISS 인덱싱
```

### 3. 질의 처리 단계
```
사용자 질문 → Query Expansion → 검색 → Reranking → LLM 생성 → 응답
```

## 🔧 설정 및 환경 변수

### 필수 환경 변수 (.env)
```bash
OPENAI_API_KEY=         # OpenAI API 키
KAKAO_REST_KEY=         # Kakao REST API 키
TMAP_API_KEY=           # TMap API 키
LANGCHAIN_API_KEY=      # LangChain 추적 키 (선택)
MONGO_URI=              # MongoDB 연결 URI (선택)
```

### RAG 설정 (rag_config.yaml)
- **Retrieval**: top_k=5, similarity search
- **Reranking**: MonoT5, batch_size=4
- **Embedding**: BAAI/bge-small-en-v1.5
- **Generation**: GPT-4o-mini, temperature=0.7

## 📈 성능 메트릭

### RAG 평가 지표
- **Retrieval 성능**
  - Precision@K
  - Recall@K
  - MRR (Mean Reciprocal Rank)

- **Generation 품질**
  - BLEU Score
  - ROUGE Score
  - Semantic Similarity

### 시스템 성능
- 평균 응답 시간: < 3초
- 동시 사용자 지원: 100+
- 메모리 사용량: < 2GB

## 🧪 평가 시스템

### 평가 프레임워크
- **RAGAS**: 자동화된 RAG 평가 프레임워크
- **LangSmith**: 실시간 모니터링 및 추적
- **평가 데이터셋**: 200개 실제 질문-답변 쌍

### 평가 메트릭 상세

#### 1. Faithfulness (충실도)
- 생성된 답변이 검색된 컨텍스트에 기반하는 정도
- 환각(Hallucination) 최소화 지표
- 목표 수준: > 0.7

#### 2. Factual Correctness (사실 정확도)
- F1 스코어 기반 답변 정확성
- Ground Truth와의 일치도 측정
- 목표 수준: > 0.6

### 모델 벤치마크 결과

| 모델 | 파라미터 | Faithfulness | Factual Correctness | 비고 |
|------|----------|-------------|-------------------|------|
| **GPT-4o-mini** | - | **0.6388** | **0.4881** | 프로덕션 권장 |
| Google Gemma 3-4B | 4B | 0.5984 | 0.4223 | 비용 효율적 |
| DeepSeek Chat | - | 0.5682 | 0.4400 | 균형잡힌 성능 |
| Qwen3 | 4B | 0.5090 | 0.4147 | 오픈소스 대안 |
| Exaone 3.5 | 2.4B | 0.4023 | 0.3710 | 경량 모델 |
| Kakao Kanana | 2.1B | 0.4089 | 0.3714 | 한국어 특화 |
| HyperClovaX Seed | 1.5B | 0.3661 | 0.3020 | 엣지 디바이스용 |

### 평가 데이터 생성 파이프라인

```python
# 1. 문서 청킹
chunk_size = 512
chunk_overlap = 128

# 2. QA 쌍 자동 생성
- 모델: GPT-4o-mini
- 질문 유형: 단일 홉 특정 쿼리
- 언어: 한국어
- 검증: 도메인 전문가 리뷰

# 3. 평가 실행
- 평가자 모델: GPT-4.1, Grok-3-beta
- 교차 검증으로 신뢰성 확보
```

### 평가 인프라

#### 로컬 모델 서빙
- **Ollama**: 오픈소스 모델 실행
- **LMS (Local Model Server)**: 소규모 모델 최적화
- **벡터스토어**: ChromaDB (150개 문서)
- **임베딩**: text-embedding-3-small (1536차원)

#### 평가 자동화
```bash
# 평가 노트북
- evaluate_llms.ipynb: LLM 비교 평가
- eval_ragas.ipynb: RAGAS 메트릭 계산
- local_llm_evaluations.ipynb: 로컬 모델 테스트

# 결과 파일
- eval_results_200.json: 기본 평가 결과
- eval_results_200_grok.json: Grok 평가 결과
- eval_results_200_for_four.json: 4개 모델 비교
```

### 평가 기반 인사이트

#### 성능 vs 비용 분석
1. **프로덕션 추천**: GPT-4o-mini
   - 최고 성능, API 기반 관리 용이

2. **비용 최적화**: Google Gemma 3-4B
   - 성능 대비 비용 효율적

3. **온프레미스**: Exaone 3.5
   - 자체 서버 운영시 적합

4. **엣지 컴퓨팅**: HyperClovaX Seed
   - 제한된 리소스 환경용

#### 개선 방향
- 평가 데이터셋 확장 (200 → 500개)
- 멀티턴 대화 평가 추가
- 실시간 사용자 피드백 통합
- A/B 테스트 프레임워크 구축

## 🚀 실행 방법

### 1. 데이터 준비
```bash
# 웹 크롤링
python -m src.data_processor.crawler

# QA 데이터셋 생성
python main.py generate --num-samples 200 --questions-per-content 3
```

### 2. RAG 파이프라인 검증
```bash
# 설정 검증
python main.py validate

# 파이프라인 평가
python main.py evaluate
```

### 3. 웹 애플리케이션 실행
```bash
streamlit run src/legacy/app.py
```

## 🔄 향후 개선 사항

### 단기 계획
1. **코드 리팩토링**
   - Legacy 코드 현대화
   - 모듈 간 의존성 정리
   - 테스트 코드 추가

2. **기능 개선**
   - 실시간 열차 위치 추적
   - 혼잡도 예측
   - 개인화된 경로 추천

3. **성능 최적화**
   - 캐싱 메커니즘 구현
   - 벡터 검색 속도 개선
   - 응답 시간 단축

### 장기 계획
1. **확장성**
   - 다른 도시 지하철 시스템 지원
   - 버스 시스템 통합
   - 실시간 교통 정보 연동

2. **AI 고도화**
   - Fine-tuning 모델 개발
   - 다중 모달 지원 (이미지, 음성)
   - 예측 모델 통합

3. **인프라**
   - 클라우드 배포
   - 마이크로서비스 아키텍처
   - 실시간 모니터링 시스템

## 📝 개발 노트

### 주요 이슈 및 해결
1. **토큰 제한 문제**: 청킹 전략 최적화로 해결
2. **응답 정확도**: Reranking 모델 도입으로 개선
3. **다국어 처리**: 언어별 프롬프트 템플릿 분리

### 베스트 프랙티스
- 정기적인 웹 크롤링으로 데이터 최신화
- A/B 테스트를 통한 프롬프트 개선
- 사용자 피드백 기반 지속적 개선

## 📞 문의 및 기여

프로젝트에 대한 문의나 기여는 GitHub Issues를 통해 제안해주세요.

---

*Last Updated: 2025-01-25*
*Version: 0.1.0*