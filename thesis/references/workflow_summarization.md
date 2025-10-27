# RAG 시스템 구축 및 평가 워크플로우

## 1. 문서 처리 및 벡터스토어 구축 (main_pipeline.ipynb)

### 데이터 수집 및 전처리
- `load_documents()`: datasets/final_docs 디렉토리에서 마크다운(.md) 문서를 로드
- `split_documents()`: 문서를 800 토큰 크기의 청크로 분할(100 토큰 오버랩)
- `save_splits_as_markdown()`: 분할된 청크를 타임스탬프가 포함된 하위 디렉토리에 마크다운으로 저장

### 벡터 데이터베이스 구축
- `create_embeddings()`: OpenAI text-embedding-3-small 모델 초기화
- `create_vectorstore()`: 청크를 임베딩하여 Chroma 벡터스토어에 저장
- `load_vectorstore()`: 저장된 벡터스토어를 로드

### 합성 테스트 데이터 생성
- `generate_qa_dataset()`: RAGAS를 사용하여 문서에서 질문-답변 쌍 생성
- SingleHop(70%)과 MultiHop(30%) 질문 분포로 구성
- GPT-4o를 사용하여 질문과 답변 생성

## 2. 데이터셋 번역 및 처리 (dataset_post_processing.ipynb)

### 영어 데이터셋 한국어 번역
- `translate_english_to_korean()`: 영어로 생성된 합성 데이터셋을 한국어로 번역
- GPT-4o-mini 모델을 사용하여 각 질문-답변 쌍을 한국어로 변환
- 중간 저장 기능을 통해 번역 중단 시 이어서 작업 가능

### 데이터셋 통합 및 가공
- 여러 합성 데이터셋 파일(`synthetic_qa_dataset_100.csv`, `synthetic_qa_dataset_ko_en_mixed.csv`, `synthetic_qa_dataset.csv`)을 통합
- 번역된 데이터를 JSONL 형식으로 저장하여 중간 진행 상황 보존
- 최종 번역 결과를 CSV 파일(`translated_output.csv`)로 변환하여 평가에 사용

## 3. 모델 평가 (evaluate_llms.ipynb)

### RAG 파이프라인 설정
- `create_retriever()`: 벡터스토어에서 검색기 생성 (기본 k=4)
- `create_rag_chain()`: 한국 도시철도 역무 지식 맥락에 맞춘 RAG 체인 생성
- 한국어 프롬프트 템플릿을 사용하여 도시철도 역무 도메인에 특화

### 다양한 모델 지원
- `create_llm_lms()`: LMStudio 기반 모델(exaone, qwen 등) 초기화
  - LMS 모델 관리: 모델 로드/언로드 자동화
- `create_llm_ollama()`: Ollama 기반 모델(exaone, clova, kanana 등) 초기화
  - Ollama 모델 관리: 자동 다운로드 및 실행 중인 모델 관리

### 평가 실행
- 다양한 로컬 LLM 모델에 대해 한국어 질문 세트로 평가 수행
- 모델별 결과를 JSON 파일로 저장 (예: `result_exaone_3.5_2.json`)
- LangSmith 통합을 통해 모델 성능 시각화 및 비교

## 워크플로우 요약

1. **데이터 준비 단계**
   - 마크다운 문서 로드 → 청크 분할 → 임베딩 → Chroma 벡터스토어 구축
   - RAGAS를 통한 질문-답변 합성 데이터셋 생성

2. **데이터셋 번역 단계**
   - 영어 합성 데이터셋 → GPT-4o-mini로 한국어 번역 → CSV/JSONL 저장
   - 다양한 데이터셋 통합으로 평가 데이터 풍부화

3. **모델 평가 단계**
   - 다양한 로컬 LLM 모델 설정 및 RAG 파이프라인 구성
   - 번역된 한국어 질문 세트로 각 모델 평가 실행
   - 모델별 응답 저장 및 LangSmith에 업로드하여 성능 비교 분석

## 사용된 주요 기술 및 모델

### 임베딩 및 검색
- **임베딩**: OpenAI text-embedding-3-small
- **벡터스토어**: ChromaDB
- **검색기**: 유사도 기반 검색 (k=4)

### 로컬 언어 모델
- **LMStudio 기반**: exaone-3.5-2.4b-instruct, qwen3-4b 등
- **Ollama 기반**: exaone3.5, HyperCLOVAX-SEED, kanana-nano, llama3.1 등

### 평가 및 데이터 생성
- **질문-답변 생성**: GPT-4o
- **번역**: GPT-4o-mini
- **평가 플랫폼**: LangSmith
- **합성 데이터 생성**: RAGAS

### 특징
- 모든 과정에서 중간 결과 저장을 통한 작업 연속성 확보
- 다양한 로컬 LLM 모델에 대한 통합 관리 및 자동화
- 한국어 도메인(도시철도 역무)에 특화된 질문-답변 데이터셋 구축
