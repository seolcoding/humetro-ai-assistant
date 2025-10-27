# Humetro AI Assistant

부산 도시철도(Humetro) AI 어시스턴트를 위한 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 프로젝트 개요

이 프로젝트는 부산 도시철도 이용자에게 정확한 정보를 제공하기 위한 대화형 AI 어시스턴트를 개발하는 것을 목표로 합니다. RAG 기술을 활용하여 도시철도 관련 정보를 검색하고 사용자의 질문에 답변합니다.

## 주요 기능

- 부산 도시철도 정보 웹사이트 크롤링
- 수집된 데이터를 활용한 QA 데이터셋 생성
- AutoRAG를 활용한 최적의 RAG 파이프라인 구성
- 검색 기반 질의응답 시스템

## 시작하기

### 요구사항

- Python 3.10 이상
- OpenAI API 키
- 필요한 패키지는 requirements.txt에 명시되어 있습니다.

### 설치

1. 저장소 클론

```bash
git clone https://github.com/yourusername/humetro-ai-assistant.git
cd humetro-ai-assistant
```

2. 가상환경 생성 및 활성화

```bash
python -m venv .venv
source .venv/bin/activate  # Windows의 경우: .venv\Scripts\activate
```

3. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

4. 환경 변수 설정

`.env.example` 파일을 `.env` 파일로 복사하고 필요한 API 키를 입력합니다.

```bash
cp .env.example .env
```

### 사용 방법

#### 데이터 수집 및 QA 데이터셋 생성

```bash
# 웹사이트 크롤링
python -m src.data_processor.crawler

# QA 데이터셋 생성
python main.py generate --num-samples 100 --questions-per-content 2
```

#### RAG 파이프라인 평가

```bash
# RAG 파이프라인 구성 검증
python main.py validate

# RAG 파이프라인 평가
python main.py evaluate
```

## 프로젝트 구조

```
humetro-ai-assistant/
├── assets/                 # 정적 파일(이미지 등)
├── crawl_result/           # 크롤링된 마크다운 파일
├── datasets/               # 데이터셋 파일
│   ├── final_docs/         # 최종 처리된 데이터셋
│   └── rag_evaluation/     # RAG 평가 결과
├── src/                    # 소스 코드
│   ├── common/             # 공통 유틸리티 함수
│   ├── data_processor/     # 데이터 처리 관련 코드
│   └── llm_tools/          # LLM 도구 관련 코드
├── .env.example            # 환경 변수 예제 파일
├── main.py                 # 메인 CLI 스크립트
├── README.md               # 프로젝트 설명
└── requirements.txt        # 필요한, 패키지 목록
```

## 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE)를 따릅니다.
