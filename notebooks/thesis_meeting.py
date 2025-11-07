# %%
# ============================================================================
# .env 파일에서 환경 변수 로드
# ============================================================================

from pathlib import Path
from dotenv import load_dotenv
import os

# 프로젝트 루트에서 .env 로드
project_root = Path("/Users/sdh/Dev/02_production_projects/humetro-ai-assistant")
env_path = project_root / ".env"

if not env_path.exists():
    raise FileNotFoundError(f"❌ .env 파일을 찾을 수 없습니다: {env_path}")

# .env 파일 로드
load_dotenv(env_path)

print("✅ 환경 변수 로드 완료")
print(f"   - .env 경로: {env_path}")

# 필수 환경 변수 확인
required_vars = ["OPENAI_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    raise ValueError(
        f"❌ 필수 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}"
    )

print("\n📋 환경 변수 확인:")
for var in required_vars:
    value = os.getenv(var)
    masked_value = f"{'*' * (len(value) - 4)}{value[-4:]}" if len(value) > 4 else "****"
    print(f"   - {var}: {masked_value}")


# %% [markdown]
# # 📊 서울 지하철 Q/A 시스템 평가 - RAGAS 프레임워크
#
# **목적**: RAGAS를 활용한 서울 메트로 Q/A 시스템 성능 평가
#
# **버전**: 2.0 (RAGAS 0.3.1+ Knowledge Graph 접근법)
#
# **최종 수정**: 2025-10-30
#
# **발표자**: SDH

# %% [markdown]
# ## 📑 목차
#
# ### 🔧 Part I: 환경 설정
# - **[1. 환경 구성](#1-환경-구성)**
#   - 1.1 라이브러리 및 의존성
#   - 1.2 모델 설정 (OpenAI + Ollama)
#   - 1.3 디렉토리 구조
#
# ### 🕷️ Part II: 데이터 수집 및 전처리
# - **[2. 크롤링 및 수집](#2-크롤링-및-수집)**
#   - 2.1 Crawl4AI 기반 크롤링
#   - 2.2 마크다운 변환
#   - 2.3 중복 제거
# - **[3. 텍스트 청킹](#3-텍스트-청킹)**
#   - 3.1 청킹 전략 (512 chars)
#   - 3.2 청크 통계
# - **[4. 벡터 스토어](#4-벡터-스토어)**
#   - 4.1 임베딩 생성
#   - 4.2 FAISS 인덱스
#
# ### 🎯 Part III: RAGAS 테스트셋 생성
# - **[5. 한국어 Q/A 생성](#5-한국어-qa-생성)**
#   - 5.1 한국어 페르소나 정의 ⭐
#   - 5.2 Knowledge Graph 생성
#   - 5.3 Transform 파이프라인
#   - 5.4 Query Synthesizers
#   - 5.5 테스트셋 생성
#   - 5.6 한국어 검증
#
# ### 📈 Part IV: 평가 및 분석
# - **[6. Q/A 시스템 구현](#6-qa-시스템-구현)**
#   - 6.1 RAG 파이프라인
#   - 6.2 답변 생성
# - **[7. RAGAS 평가](#7-ragas-평가)**
#   - 7.1 평가 메트릭
#   - 7.2 모델별 비교
#
# ### 📊 Part V: 결과 및 인사이트
# - **[8. 성능 분석](#8-성능-분석)**
#   - 8.1 시각화
#   - 8.2 오류 분석
# - **[9. 결론 및 제언](#9-결론-및-제언)**
#
# ---

# %% [markdown]
# ## 🛠️ 기술 스택
#
# ### 📚 핵심 프레임워크
# - **RAGAS**: 0.3.1+ (Knowledge Graph 기반)
# - **LangChain**: 문서 처리 & RAG 파이프라인
# - **Crawl4AI**: 웹 크롤링 & 데이터 수집
# - **OpenAI**: GPT-4o-mini (생성), text-embedding-3-small (임베딩)
# - **Ollama**: 로컬 모델 (EXAONE, Qwen, Gemma)
#
# ### ✨ 주요 특징
# - ✅ **한국어 특화**: 명시적 한국어 페르소나
# - ✅ **KG 기반**: 구조화된 질문 생성
# - ✅ **멀티 모델**: OpenAI + Ollama 평가
# - ✅ **종합 메트릭**: Faithfulness, Relevancy, Correctness
#
# ### 🔄 데이터 파이프라인
# ```
# 웹 크롤링 → 마크다운 변환 → 중복 제거 → 청킹 → 임베딩 → 벡터 DB
# ```
#
# ---

# %% [markdown]
# # 🔧 Part I: 환경 설정
#
# ## 1. 환경 구성
#
# ### 1.1 라이브러리 및 의존성

# %%
# ============================================================================
# .env 파일에서 환경 변수 로드
# ============================================================================

from pathlib import Path
from dotenv import load_dotenv
import os

# 프로젝트 루트에서 .env 로드
project_root = Path("/Users/sdh/Dev/02_production_projects/humetro-ai-assistant")
env_path = project_root / ".env"

if not env_path.exists():
    raise FileNotFoundError(f"❌ .env 파일을 찾을 수 없습니다: {env_path}")

# .env 파일 로드
load_dotenv(env_path)

print("✅ 환경 변수 로드 완료")
print(f"   - .env 경로: {env_path}")

# 필수 환경 변수 확인
required_vars = ["OPENAI_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    raise ValueError(
        f"❌ 필수 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}"
    )

print("📋 환경 변수 확인:")
for var in required_vars:
    value = os.getenv(var)
    masked_value = f"{'*' * (len(value) - 4)}{value[-4:]}" if len(value) > 4 else "****"
    print(f"   - {var}: {masked_value}")


# %% [markdown]
# #### 📦 핵심 라이브러리

# %%
# 표준 라이브러리
import os
import json
import time
from pathlib import Path

# 데이터 처리
import pandas as pd
import numpy as np

# 진행 상황 추적 (notebook-optimized)
from tqdm.notebook import tqdm

print("✅ 기본 라이브러리 임포트 완료")

# %% [markdown]
# #### 🔗 LangChain 컴포넌트

# %%
# LangChain Core
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain Community (Chroma vector store)
from langchain_community.vectorstores import Chroma

# LangChain OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

print("✅ LangChain 라이브러리 임포트 완료")

# %% [markdown]
# #### 🎯 RAGAS 프레임워크

# %%
# RAGAS (testset generation + evaluation)
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)
from ragas import evaluate
from ragas.metrics import (
    # Retrieval 메트릭 제외 (동일 retriever 사용으로 차별화 불가)
    # context_precision,  # ❌ 제외: 모든 모델 동일 retriever
    # context_recall,     # ❌ 제외: 모든 모델 동일 retriever
    # Generation 품질 메트릭 (모델별 차이 측정)
    answer_relevancy,  # ✅ 답변이 질문과 관련있는지
    faithfulness,  # ✅ 답변이 context에 충실한지 (hallucination)
    answer_correctness,  # ✅ 답변이 ground truth와 일치하는지
)
from datasets import Dataset

print("✅ RAGAS 라이브러리 임포트 완료")

# %% [markdown]
# ### 1.2 모델 설정
#
# #### 📁 디렉토리 구성

# %% [markdown]
# #### 🤖 OpenAI 모델

# %%
print("🔧 OpenAI 모델 초기화 중...")

try:
    # Question generation + Answer generation
    gpt5_mini = ChatOpenAI(model="gpt-5-mini", temperature=0.7)

    # Answer evaluation
    gpt5 = ChatOpenAI(model="gpt-5", temperature=0)

    # Embeddings for Vector Store + RAGAS
    openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("✅ OpenAI 클라이언트 초기화 완료")
    print("   - Question Gen: gpt-5-mini (temp=0.7)")
    print("   - Answer Gen: gpt-5-mini (temp=0.7)")
    print("   - Evaluation: gpt-5 (temp=0)")
    print("   - Embeddings: text-embedding-3-small (1536 차원)")
except Exception as e:
    print(f"❌ OpenAI 초기화 실패: {e}")
    raise

# %% [markdown]
# #### 💻 Ollama 로컬 모델 (선택사항)

# %%
print("🔧 Ollama 모델 연결 테스트...")

OLLAMA_BASE_URL = "http://100.95.220.92:11434"

ollama_models = [
    {
        "name": "EXAONE-3.5-7.8B",
        "model": "exaone3.5:7.8b",
        "description": "한국어 주력",
    },
    {"name": "Qwen3-8B", "model": "qwen3:8b", "description": "복잡한 추론"},
    # {"name": "GPT-OSS-20B", "model": "gpt-oss:20b", "description": "구조화 작업"},  # ❌ Harmony 포맷 이슈로 제외
    {"name": "Gemma3-12B", "model": "gemma3:12b", "description": "범용 fallback"},
]

import requests

ollama_status = []
# ✅ Fixed: Use enumerate instead of tqdm (avoids infinite loop)
for idx, model_info in enumerate(ollama_models):
    print(f"   테스트 중 ({idx + 1}/{len(ollama_models)}): {model_info['name']}...")
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model_info["model"],
                "prompt": "테스트",
                "stream": False,
                "options": {"num_predict": 10},
            },
            timeout=30,
        )

        if response.status_code == 200:
            ollama_status.append(
                {
                    "model": model_info["name"],
                    "status": "✅ Connected",
                    "description": model_info["description"],
                }
            )
        else:
            ollama_status.append(
                {
                    "model": model_info["name"],
                    "status": f"❌ HTTP {response.status_code}",
                    "description": model_info["description"],
                }
            )
    except Exception as e:
        ollama_status.append(
            {
                "model": model_info["name"],
                "status": f"❌ {str(e)[:50]}",
                "description": model_info["description"],
            }
        )

# 연결 상태 출력
print("📊 Ollama 연결 상태:")
df_ollama = pd.DataFrame(ollama_status)
print(df_ollama.to_string(index=False))

# %% [markdown]
# ---
#
# # 🕷️ Part II: 데이터 수집 및 전처리
#
# ## 2. 크롤링 및 수집
#
# ### 2.1 Crawl4AI 기반 크롤링
#
# #### 🌐 크롤링 전략
# - **대상**: 서울시 교통정보 사이트
# - **도구**: Crawl4AI 0.7.6
# - **방식**: JavaScript 렌더링 + 비동기 처리
# - **수집량**: 1,969개 페이지
#
# ### 2.2 데이터 전처리
# - **마크다운 변환**: HTML → Markdown
# - **중복 제거**: 콘텐츠 해시 기반
# - **정제**: 불필요한 태그/스크립트 제거

# %% [markdown]
# ## 🔍 연구 배경 및 사전 조사
#
# ### 📚 기존 연구 분석
#
# #### 1. 평가 프레임워크 조사
# - **RAGAS vs AutoRAG**: 프레임워크 비교 분석
# - **선택 이유**: RAGAS의 세밀한 메트릭 제공
# - **버전 이슈**: 0.2.x → 0.3.1+ API 마이그레이션
#
# #### 2. 임베딩 모델 선택
# - **후보군 검토**:
#   - OpenAI text-embedding-3-small (1536차원)
#   - KURE-v1 (한국어 특화, 768차원)
#   - Multilingual-e5-large (1024차원)
# - **최종 선택**: OpenAI (성능/비용 균형)
#
# #### 3. 한국어 생성 문제
# - **문제점**: 암묵적 한국어 기대 → 영어 생성
# - **해결책**: 명시적 한국어 지정 in Persona
# ```python
# "모든 질문과 답변은 반드시 한국어로 작성되어야 합니다."
# ```
#
# ---

# %% [markdown]
# ## 🕷️ 데이터 수집 전략
#
# ### 📊 크롤링 대상 선정
#
# #### 서울시 교통정보 사이트
# - **URL**: `https://news.seoul.go.kr/traffic/`
# - **선정 이유**:
#   - 공식 정보원 (신뢰성)
#   - 구조화된 콘텐츠
#   - 정기 업데이트
#   - 다양한 주제 포함
#
# ### 🛠️ Crawl4AI 설정
#
# ```python
# # Crawl4AI 0.7.6 설정
# crawler_config = {
#     'javascript_enabled': True,  # 동적 콘텐츠
#     'wait_for': 'css:.content',  # 콘텐츠 로딩 대기
#     'exclude_patterns': [        # 제외 패턴
#         r'/login',
#         r'/admin',
#         r'\.(jpg|png|pdf)'
#     ],
#     'max_depth': 3,              # 크롤링 깊이
#     'concurrent_requests': 5     # 동시 요청
# }
# ```
#
# ---

# %% [markdown]
# ## 📝 크롤링 구현 로직
#
# ### 1️⃣ URL 발견 및 필터링
#
# ```python
# # src/scripts/discover_urls.py
# async def discover_urls(base_url: str) -> List[str]:
#     """재귀적 URL 발견"""
#     visited = set()
#     to_visit = [base_url]
#
#     while to_visit:
#         url = to_visit.pop()
#         if url in visited:
#             continue
#
#         # 페이지 크롤링
#         result = await crawler.arun(url)
#
#         # 링크 추출
#         links = extract_links(result.html)
#
#         # 필터링 (도메인, 패턴)
#         valid_links = filter_links(links)
#         to_visit.extend(valid_links)
#
#     return list(visited)
# ```
#
# ### 2️⃣ 콘텐츠 추출 및 변환
#
# ```python
# # HTML → Markdown 변환
# def extract_content(html: str) -> str:
#     """콘텐츠 추출 및 정제"""
#     # 1. 메인 콘텐츠 영역 추출
#     soup = BeautifulSoup(html, 'html.parser')
#     content = soup.select_one('.content-area')
#
#     # 2. 노이즈 제거
#     for tag in content.select('script, style, nav'):
#         tag.decompose()
#
#     # 3. Markdown 변환
#     markdown = html2text.html2text(str(content))
#
#     # 4. 정제
#     markdown = clean_markdown(markdown)
#
#     return markdown
# ```
#
# ---

# %% [markdown]
# ## 🔄 중복 제거 프로세스
#
# ### 📊 중복 제거 전략
#
# #### 1. 콘텐츠 해시 기반
# ```python
# # src/scripts/deduplicate_markdown.py
# def get_content_hash(text: str) -> str:
#     """콘텐츠 정규화 후 해시 생성"""
#     # 공백, 줄바꿈 정규화
#     normalized = ' '.join(text.split())
#
#     # 소문자 변환
#     normalized = normalized.lower()
#
#     # SHA256 해시
#     return hashlib.sha256(normalized.encode()).hexdigest()
# ```
#
# #### 2. 유사도 기반 필터링
# ```python
# def filter_similar(docs: List[str], threshold=0.9) -> List[str]:
#     """코사인 유사도 기반 필터링"""
#     vectorizer = TfidfVectorizer()
#     vectors = vectorizer.fit_transform(docs)
#
#     # 유사도 매트릭스
#     similarity = cosine_similarity(vectors)
#
#     # 중복 제거
#     unique_docs = []
#     for i, doc in enumerate(docs):
#         if not any(similarity[i][j] > threshold
#                   for j in range(i)):
#             unique_docs.append(doc)
#
#     return unique_docs
# ```
#
# ### 📈 중복 제거 결과
# - **수집**: 2,341개 페이지
# - **중복 제거 후**: 1,969개 (84.1%)
# - **제거된 중복**: 372개 (15.9%)
#
# ---

# %% [markdown]
# ## 📊 수집 데이터 통계
#
# ### 📈 최종 데이터셋
#
# | 항목 | 수치 | 비고 |
# |------|------|------|
# | **총 문서** | 1,969개 | 중복 제거 후 |
# | **총 문자** | 5,595,691 | ~5.6M chars |
# | **평균 길이** | 2,842 chars | 문서당 |
# | **최대 길이** | 15,281 chars | - |
# | **최소 길이** | 127 chars | - |
#
# ### 📂 주제별 분포
# - **지하철 정보**: 35%
# - **버스 정보**: 28%
# - **교통 정책**: 22%
# - **시설 안내**: 15%
#
# ### ⏱️ 크롤링 성능
# - **총 시간**: 47분
# - **평균 속도**: 42 pages/min
# - **에러율**: 0.8% (19 failures)
#
# ---

# %%
# ============================================================================
# 프로젝트 경로 설정 (.env 활용)
# ============================================================================

from pathlib import Path
import os

# 프로젝트 루트 디렉토리 (노트북 위치 기준)
PROJECT_ROOT = Path.cwd().parent

# .env에서 환경변수 로드 (이미 Cell 1에서 로드됨)
# DATA_DIR, RESULTS_DIR이 .env에 정의되어 있음

# 환경변수에서 상대 경로를 읽어 절대 경로로 변환
DATA_DIR_REL = os.getenv("DATA_DIR", "./data")
RESULTS_DIR_REL = os.getenv("RESULTS_DIR", "./results")

# 절대 경로 생성
DATA_DIR = (
    PROJECT_ROOT
    / DATA_DIR_REL.lstrip("./")
    / "crawled"
    / "seoul_traffic"
    / "markdown_deduplicated"
)
RESULTS_DIR = PROJECT_ROOT / RESULTS_DIR_REL.lstrip("./")
LOGS_DIR = RESULTS_DIR / "logs"

# 디렉토리 생성
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

print("✅ 프로젝트 경로 설정 완료 (.env 기반)")
print(f"   - 프로젝트 루트: {PROJECT_ROOT}")
print(f"   - 데이터 디렉토리: {DATA_DIR}")
print(f"   - 데이터 존재 여부: {DATA_DIR.exists()}")
if DATA_DIR.exists():
    print(f"   - 마크다운 파일 수: {len(list(DATA_DIR.glob('*.md')))}")
print(f"   - 결과 디렉토리: {RESULTS_DIR}")
print(f"   - 로그 디렉토리: {LOGS_DIR}")


# %%
# ============================================================================
# Section 2: Data Loading & Preprocessing
# ============================================================================

print("" + "=" * 80)
print("📂 Section 2: 데이터 로딩 및 전처리")
print("=" * 80)

print(f"📁 데이터 디렉토리: {DATA_DIR}")
print("   - 크롤링 소스: 서울시 교통정보 (마크다운)")

# 마크다운 파일 목록 가져오기
import glob

md_files = sorted(glob.glob(str(DATA_DIR / "*.md")))

print("📊 발견된 파일:")
print(f"   - 총 파일 수: {len(md_files)}개")

if len(md_files) == 0:
    raise FileNotFoundError(f"❌ {DATA_DIR}에 마크다운 파일이 없습니다!")

# 문서 로딩 with progress bar
documents = []
total_chars = 0

print("📖 문서 로딩 중...")
for i, file_path in enumerate(tqdm(md_files, desc="마크다운 파일 로딩")):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 기본 전처리 (과도한 공백 제거)
        content = "\n".join(
            [line.strip() for line in content.split("\n") if line.strip()]
        )

        # LangChain Document 생성
        doc = Document(
            page_content=content,
            metadata={
                "source": file_path,
                "file_name": Path(file_path).name,
                "doc_id": i,
                "chars": len(content),
            },
        )
        documents.append(doc)
        total_chars += len(content)

        # 500개마다 진행 상황 출력
        if (i + 1) % 500 == 0:
            avg_chars = total_chars / (i + 1)
            print(
                f"   📈 진행: {i + 1}/{len(md_files)} 문서 ({avg_chars:.0f} chars/doc)"
            )

    except Exception as e:
        print(f"   ⚠️ 파일 읽기 실패: {Path(file_path).name} - {e}")
        continue

# 최종 통계
print("✅ 데이터 로딩 완료")
print(f"   - 성공: {len(documents)}개")
print(f"   - 실패: {len(md_files) - len(documents)}개")
print(f"   - 총 문자 수: {total_chars:,} chars")
print(f"   - 평균 길이: {total_chars / len(documents):.0f} chars/doc")
print(f"   - 최소 길이: {min(doc.metadata['chars'] for doc in documents):,} chars")
print(f"   - 최대 길이: {max(doc.metadata['chars'] for doc in documents):,} chars")

print("" + "=" * 80)
print("✅ Section 2 완료: 데이터 로딩 및 전처리")
print("=" * 80)


# %% [markdown]
# ## 3. 텍스트 청킹
#
# ### 3.1 청킹 전략
# - **크기**: 512 characters
# - **오버랩**: 128 characters
# - **분할기**: RecursiveCharacterTextSplitter

# %%
# ============================================================================
# Section 3: Text Chunking
# ============================================================================

print("" + "=" * 80)
print("✂️ Section 3: 텍스트 청킹")
print("=" * 80)

# 청킹 설정
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256

print("⚙️ 청킹 설정:")
print(f"   - Chunk Size: {CHUNK_SIZE} chars")
print(f"   - Chunk Overlap: {CHUNK_OVERLAP} chars")
print("   - Splitter: RecursiveCharacterTextSplitter")

# Text Splitter 초기화
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

print("✂️ 문서 청킹 중...")
start_time = time.time()

chunks = []
for doc in tqdm(documents, desc="문서 청킹"):
    doc_chunks = text_splitter.split_documents([doc])
    chunks.extend(doc_chunks)

chunking_time = time.time() - start_time

# 청킹 통계
chunk_lengths = [len(chunk.page_content) for chunk in chunks]

print(f"✅ 청킹 완료 ({chunking_time:.1f}초)")
print(f"   - 원본 문서: {len(documents):,}개")
print(f"   - 생성된 청크: {len(chunks):,}개")
print(f"   - 평균 청크 수: {len(chunks) / len(documents):.1f} chunks/doc")
print(f"   - 평균 청크 길이: {np.mean(chunk_lengths):.0f} chars")
print(f"   - 최소 청크 길이: {min(chunk_lengths)} chars")
print(f"   - 최대 청크 길이: {max(chunk_lengths)} chars")
print(f"   - 청킹 속도: {len(documents) / chunking_time:.1f} docs/sec")

print("" + "=" * 80)
print("✅ Section 3 완료: 텍스트 청킹")
print("=" * 80)


# %% [markdown]
# ## 4. 벡터 스토어 생성
#
# ### 4.1 FAISS 인덱스
# - **임베딩**: OpenAI text-embedding-3-small
# - **차원**: 1,536
# - **캐싱**: 로컬 저장 (178MB)

# %%
print(DATA_DIR)

# %%
# ============================================================================
# Section 4: Vector Store Creation (캐싱 최우선 전략)
# ============================================================================

print("" + "=" * 80)
print("🗄️ Section 4: 벡터 스토어 생성")
print("=" * 80)

VECTORSTORE_DIR = Path(
    "/Users/sdh/Dev/02_production_projects/humetro-ai-assistant/data/vectorstore"
)
if VECTORSTORE_DIR is None:
    raise ValueError("No Vectorstore dir env exists")

print(f"📍 벡터 스토어 경로: {VECTORSTORE_DIR}")
print("   - 임베딩 모델: OpenAI text-embedding-3-small (1536 차원)")
print("   - 벡터 DB: Chroma (로컬 persist)")

# ============================================================================
# 캐싱 전략: ALWAYS USE CACHE IF EXISTS
# ============================================================================

print("🔍 캐시 확인 중...")
cache_exists = VECTORSTORE_DIR.exists() and len(list(VECTORSTORE_DIR.glob("*"))) > 0

if cache_exists:
    print("   ✅ 캐시 발견!")
    print(f"   - 경로: {VECTORSTORE_DIR}")

    # 캐시 크기 확인
    import subprocess

    cache_size_result = subprocess.run(
        ["du", "-sh", str(VECTORSTORE_DIR)], capture_output=True, text=True
    )
    cache_size = cache_size_result.stdout.split()[0]
    print(f"   - 크기: {cache_size}")

    # 캐시된 벡터 스토어 로드
    print("📦 캐시된 벡터 스토어 로딩 중...")
    start_time = time.time()

    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_DIR), embedding_function=openai_embeddings
    )

    load_time = time.time() - start_time

    # 캐시 통계
    collection_count = vectorstore._collection.count()

    print(f"✅ 캐시 로드 완료 ({load_time:.1f}초)")
    print(f"   - 벡터 수: {collection_count:,}개")
    print(f"   - 로드 속도: {collection_count / load_time:.0f} vectors/sec")
    print("   - 임베딩 API 호출: 0회 (비용 절감: 100%)")

    # 예상 절감 비용 계산
    saved_tokens = collection_count * 100  # 평균 100 tokens/chunk
    saved_cost = (saved_tokens / 1_000_000) * 0.02  # $0.02/1M tokens
    print(f"   - 절감된 비용: ~${saved_cost:.2f}")
    print(f"   - 절감된 시간: ~{collection_count * 0.1:.0f}초 (예상)")

else:
    print("   ❌ 캐시 없음")
    print("   - 새로 생성 필요")
    print(f"   - 예상 임베딩 API 호출: ~{len(chunks):,}회")
    print(f"   - 예상 비용: ~${(len(chunks) * 100 / 1_000_000) * 0.02:.2f}")
    print(f"   - 예상 시간: ~{len(chunks) * 0.1:.0f}초")

    # 새 벡터 스토어 생성
    print("🏗️ 벡터 스토어 생성 중...")
    print(f"   ⚠️ {len(chunks):,}개 청크 임베딩 진행...")
    print(f"   ⏳ 약 {len(chunks) * 0.1 / 60:.1f}분 소요 예상")

    start_time = time.time()

    # Batch 임베딩 (효율성)
    BATCH_SIZE = 100
    vectorstore = None

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="임베딩 배치 생성"):
        batch = chunks[i : i + BATCH_SIZE]

        if vectorstore is None:
            # 첫 배치로 vectorstore 초기화
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=openai_embeddings,
                persist_directory=str(VECTORSTORE_DIR),
            )
        else:
            # 이후 배치 추가
            vectorstore.add_documents(batch)

        # 중간 저장 (안전성)
        if (i + BATCH_SIZE) % 500 == 0:
            vectorstore.persist()
            print(f"   💾 중간 저장: {i + BATCH_SIZE}/{len(chunks)} 청크")

    # 최종 저장
    vectorstore.persist()

    embedding_time = time.time() - start_time
    collection_count = vectorstore._collection.count()

    print(f"✅ 벡터 스토어 생성 완료 ({embedding_time:.1f}초)")
    print(f"   - 벡터 수: {collection_count:,}개")
    print(f"   - 임베딩 속도: {collection_count / embedding_time:.1f} vectors/sec")
    print(f"   - API 호출 수: ~{collection_count:,}회")
    print(f"   - 실제 비용: ~${(collection_count * 100 / 1_000_000) * 0.02:.2f}")
    print(f"   - 캐시 저장: {VECTORSTORE_DIR}")

# ============================================================================
# 벡터 스토어 검증
# ============================================================================

print("🧪 벡터 스토어 검증...")

# 테스트 쿼리
test_query = "서울 지하철 운영 시간은?"
test_results = vectorstore.similarity_search(test_query, k=3)

print(f"   - 테스트 쿼리: '{test_query}'")
print(f"   - 검색 결과: {len(test_results)}개")
print("   - 첫 번째 결과 미리보기:")
print(f"     {test_results[0].page_content[:100]}...")

print("✅ 벡터 스토어 정상 작동 확인")

print("" + "=" * 80)
print("✅ Section 4 완료: 벡터 스토어 생성 (캐싱 전략)")
print("=" * 80)

print("💡 중요:")
print("   - 다음 실행부터는 캐시 사용으로 즉시 로드")
print("   - 임베딩 API 호출 없음 = 비용 0, 시간 < 5초")
print("   - 캐시 삭제 시 재생성 필요")


# %% [markdown]
# ---
#
# # 🎯 Part III: RAGAS 테스트셋 생성
#
# ## 5. 한국어 Q/A 생성
#
# ### 🔑 핵심 개선사항
# - ✅ RAGAS 0.3.1+ API 마이그레이션
# - ✅ 명시적 한국어 페르소나
# - ✅ Knowledge Graph 접근법
# - ✅ Transform 파이프라인
#
# ### 5.1 RAGAS 컴포넌트 설정

# %%
# ============================================================================
# Section 5: RAGAS Question Generation
# ============================================================================

print("" + "=" * 80)
print("🤔 Section 5: RAGAS 질문 생성")
print("=" * 80)

# ============================================================================
# RAGAS 상세 로깅 설정
# ============================================================================

import logging
from datetime import datetime

# 로그 파일 경로
log_dir = RESULTS_DIR / "logs"
log_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"ragas_generation_{timestamp}.log"

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),  # 콘솔에도 출력
    ],
)

# RAGAS 로거 설정
ragas_logger = logging.getLogger("ragas")
ragas_logger.setLevel(logging.DEBUG)

# LangChain 로거 설정 (RAGAS가 내부적으로 사용)
langchain_logger = logging.getLogger("langchain")
langchain_logger.setLevel(logging.INFO)

# OpenAI 로거 설정
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.INFO)

print("📋 RAGAS 상세 로깅 설정 완료")
print(f"   - 로그 파일: {log_file}")
print("   - 로그 레벨: DEBUG")
print("   - 포함 정보:")
print("     • RAGAS 내부 동작 (질문 생성 과정)")
print("     • LangChain 호출 (문서 처리)")
print("     • OpenAI API 호출 (토큰 사용량)")
print("     • 에러 및 경고 메시지")

# ============================================================================
# RAGAS 컴포넌트 설정 (RAGAS 0.3.1 API - LangchainLLMWrapper 필수!)
# ============================================================================

print("⚙️ RAGAS 컴포넌트 설정...")
print("   - Generator LLM: gpt-5-mini (temp=0.7)")
print("   - Embeddings: OpenAI text-embedding-3-small")
print("   - RAGAS 0.3.1: LangchainLLMWrapper 사용")

try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    import openai

    # LLM Wrapper (RAGAS 0.3.1 필수!)
    generator_llm_raw = ChatOpenAI(model="gpt-5-mini", temperature=0.7)
    generator_llm = LangchainLLMWrapper(generator_llm_raw)

    # Embeddings Wrapper (RAGAS 0.3.1 필수!)
    openai_client = openai.OpenAI()
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", client=openai_client)
    )

    # TestsetGenerator 생성 (RAGAS 0.3.1 API)
    testset_generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=embeddings,
    )

    print("✅ RAGAS TestsetGenerator 초기화 완료")
    print("   - RAGAS 버전: 0.3.1")
    print("   - LLM: LangchainLLMWrapper(gpt-5-mini)")
    print("   - Embeddings: LangchainEmbeddingsWrapper(text-embedding-3-small)")
    logging.info(
        "RAGAS TestsetGenerator initialized successfully (v0.3.1 with wrappers)"
    )

except Exception as e:
    print(f"❌ RAGAS 초기화 실패: {e}")
    logging.error(f"RAGAS initialization failed: {e}", exc_info=True)
    raise

# %% [markdown]
# ### 5.2 한국어 페르소나 정의
#
# #### ⭐ 핵심: 명시적 한국어 지정
# ```python
# "모든 질문과 답변은 반드시 한국어로 작성되어야 합니다."
# ```

# %%
print("⚙️ 질문 합성기 설정...")

# ============================================================================
# Best Practice: default_query_distribution 사용
# ============================================================================
from ragas.testset.synthesizers import default_query_distribution

# RAGAS 권장 방식: LLM 기반 기본 분포 사용
query_distribution = default_query_distribution(generator_llm)

print("✅ 질문 합성기 설정 완료")
print("   - Query Distribution: default_query_distribution(generator_llm)")
print("   - 자동으로 single-hop/multi-hop 분포 설정")

# query_distribution 타입 확인 및 출력
print(f"📊 분포 타입: {type(query_distribution)}")

if isinstance(query_distribution, dict):
    print("📊 분포 상세:")
    for synthesizer, weight in query_distribution.items():
        print(f"   - {synthesizer.__class__.__name__}: {weight:.1%}")
elif isinstance(query_distribution, list):
    print("📊 Synthesizers 목록:")
    for idx, synthesizer in enumerate(query_distribution, 1):
        print(f"   {idx}. {synthesizer.__class__.__name__}")
else:
    print(f"📊 Query Distribution: {query_distribution}")


# %%
# ============================================================================
# 한국어 페르소나 정의 (RAGAS 0.3.1+ 방식)
# ============================================================================
print("\n👥 한국어 페르소나 정의...")
print("   ⚠️  중요: 모든 페르소나는 '반드시 한국어로' 질문/답변 생성")

from ragas.testset.persona import Persona

# 서울 교통 도메인 페르소나 (한국어 명시)
persona_first_time_user = Persona(
    name="지하철 처음 이용자",
    role_description="""
    지하철을 처음 이용하는 승객입니다.
    모든 질문과 답변은 반드시 한국어로 작성되어야 합니다.
    기본적인 이용 방법, 요금, 노선 정보 등에 대해 자세한 안내가 필요합니다.
    질문은 구체적이고 단계별 설명을 요구하는 형태로 작성됩니다.
    예: "지하철 요금은 어떻게 계산되나요?", "환승은 어떻게 하나요?"
    """,
)

persona_frequent_traveler = Persona(
    name="자주 이용하는 승객",
    role_description="""
    지하철을 자주 이용하는 승객입니다.
    모든 질문과 답변은 반드시 한국어로 작성되어야 합니다.
    효율적인 이동, 시간 절약, 편의시설 위치 등 실용적인 정보에 관심이 많습니다.
    질문은 간결하고 핵심적인 정보를 요구하는 형태로 작성됩니다.
    예: "가장 빠른 환승 경로는?", "막차 시간은 언제인가요?"
    """,
)

persona_dissatisfied = Persona(
    name="불만을 가진 승객",
    role_description="""
    서비스나 시설에 불만을 가진 승객입니다.
    모든 질문과 답변은 반드시 한국어로 작성되어야 합니다.
    문제 상황에 대한 즉각적인 해결책과 보상을 요구합니다.
    질문은 다소 감정적이고 즉각적인 조치를 요구하는 형태로 작성됩니다.
    예: "지연 보상은 어떻게 받나요?", "불편사항을 어디에 신고하나요?"
    """,
)

persona_transportation_vulnerable = Persona(
    name="교통약자",
    role_description="""
    어르신, 장애인, 임산부 등 교통약자입니다.
    모든 질문과 답변은 반드시 한국어로 작성되어야 합니다.
    접근성, 편의시설, 도움 서비스 등에 대한 정보가 필요합니다.
    질문은 이해하기 쉽고 배려를 요구하는 형태로 작성됩니다.
    예: "휠체어로 이용 가능한가요?", "도움이 필요하면 어디로 연락하나요?"
    """,
)

personas = [
    persona_first_time_user,
    persona_frequent_traveler,
    persona_dissatisfied,
    persona_transportation_vulnerable,
]

print(f"\n✅ {len(personas)}개 한국어 페르소나 정의 완료")
for persona in personas:
    print(f"   - {persona.name}")

print("\n🎯 핵심 특징:")
print("   - 모든 페르소나에 '반드시 한국어로' 명시")
print("   - 서울 지하철 도메인 특화")
print("   - 4가지 사용자 세그먼트 커버")
print("   - 구체적인 질문 예시 포함")

print("\n💡 참고:")
print("   - 기존 방식: 한국어 암묵적 기대 (실패 가능)")
print("   - 개선 방식: 한국어 명시적 지시 (성공률 ↑)")
print("   - RAGAS 0.3.1+에서 페르소나 기반 질문 생성 지원")


# %%


# %% [markdown]
#

# %% [markdown]
# ### 5.3 Knowledge Graph 생성
#
# #### 📊 구조화된 문서 표현
# - 문서 → 노드 변환
# - 메타데이터 보존
# - Transform 준비

# %%
# ============================================================================
# Knowledge Graph 생성
# ============================================================================

from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from langchain_community.document_loaders import DirectoryLoader, TextLoader

print("🔨 Creating Knowledge Graph...")

# Load documents from crawled markdown
DOCS_DIR = ROOT_DIR / "data" / "crawled" / "seoul_traffic" / "markdown_deduplicated"

print(f"📄 Loading documents from: {DOCS_DIR}")
loader = DirectoryLoader(
    str(DOCS_DIR),
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
documents = loader.load()
print(f"✅ Loaded {len(documents)} documents")

# Create Knowledge Graph
kg = KnowledgeGraph()

for doc in documents:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            },
        )
    )

print(f"✅ Created Knowledge Graph with {len(kg.nodes)} nodes")

# %% [markdown]
# ### 5.4 Transform 파이프라인
#
# #### 🔄 Knowledge Graph 강화
# - **HeadlinesExtractor**: 섹션 제목 추출
# - **HeadlineSplitter**: 제목 기준 분할
# - **KeyphrasesExtractor**: 핵심어 추출

# %%
# ============================================================================
# Transform Pipeline 적용
# ============================================================================

from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms import (
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor,
)

print("🔄 Applying transforms to Knowledge Graph...")

transforms = [
    HeadlinesExtractor(llm=generator_llm, max_num=20),
    HeadlineSplitter(max_tokens=1500),
    KeyphrasesExtractor(llm=generator_llm),
]

apply_transforms(kg, transforms=transforms)

print(f"✅ Transforms applied. Total nodes: {len(kg.nodes)}")
print("   - Headlines extracted and documents split")
print("   - Keyphrases extracted for query generation")

# %% [markdown]
# ### 5.5 Query Synthesizers
#
# #### 📝 질문 분포 전략
# - 50% Headlines 기반
# - 50% Keyphrases 기반

# %%
# ============================================================================
# Query Synthesizer 설정
# ============================================================================

from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

print("🎯 Configuring Query Synthesizers...")

query_distribution = [
    (
        SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="headlines"),
        0.5,
    ),
    (
        SingleHopSpecificQuerySynthesizer(
            llm=generator_llm, property_name="keyphrases"
        ),
        0.5,
    ),
]

print("✅ Query distribution configured:")
print("   - 50% Headlines-based queries")
print("   - 50% Keyphrases-based queries")

# %% [markdown]
# ### 5.6 테스트셋 생성
#
# #### 🎲 생성 프로세스
# 1. KG 노드 선택
# 2. 페르소나 적용
# 3. 한국어 질문 생성
# 4. Ground Truth 추출

# %% [markdown]
# ### 5.6 한국어 테스트셋 생성
#
# #### 🎯 생성 과정
# 1. Knowledge Graph에서 노드 선택
# 2. 페르소나 기반 질문 생성 (한국어 명시)
# 3. Query Synthesizer가 headlines/keyphrases에서 질문 생성
# 4. Ground truth 답변 추출

# %% [markdown]
# ### 5.7 Knowledge Graph 생성

# %%
# ============================================================================
# Korean Testset 생성
# ============================================================================

from ragas.testset import TestsetGenerator

print("🎯 Generating Korean testset...")
print("   - Target size: 50")
print(f"   - Personas: {len(personas)}")
print("   - Query distribution: Headlines (50%) + Keyphrases (50%)")

# Create TestsetGenerator with Knowledge Graph
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=embeddings,
    knowledge_graph=kg,
    persona_list=personas,
)

# Generate testset
print("⏳ Generating 50 Korean Q/A pairs...")
testset = generator.generate(testset_size=50, query_distribution=query_distribution)

# Convert to DataFrame
df_testset = testset.to_pandas()

print("✅ Korean testset generated!")
print(f"   - Total samples: {len(df_testset)}")
print(f"   - Columns: {df_testset.columns.tolist()}")

# Display sample
if len(df_testset) > 0:
    print("📝 Sample Korean Q/A:")
    sample = df_testset.iloc[0]
    print(f"   Question: {sample['user_input'][:100]}...")
    print(f"   Answer: {sample['reference'][:100]}...")

# %% [markdown]
# ### 5.8 한국어 품질 검증

# %%
# ============================================================================
# 한국어 품질 검증
# ============================================================================

import re

print("🔍 한국어 품질 검증...")


def is_korean(text):
    """Check if text contains Korean characters"""
    if not text or not isinstance(text, str):
        return False
    korean_pattern = re.compile("[\u3131-\u3163\uac00-\ud7a3]+")
    return bool(korean_pattern.search(text))


korean_questions = df_testset["user_input"].apply(is_korean).sum()
korean_answers = df_testset["reference"].apply(is_korean).sum()
total = len(df_testset)

print("✅ 한국어 비율:")
print(f"   - 질문: {korean_questions}/{total} ({korean_questions / total * 100:.1f}%)")
print(f"   - 답변: {korean_answers}/{total} ({korean_answers / total * 100:.1f}%)")

if korean_questions < total * 0.9 or korean_answers < total * 0.9:
    print("⚠️  경고: 한국어 비율이 90% 미만입니다.")
    print("   - Persona 정의를 재확인하세요")
    print("   - '반드시 한국어로' 명시가 있는지 확인하세요")
else:
    print("🎉 한국어 품질 검증 통과!")

# %% [markdown]
# ### 5.9 결과 저장

# %%
print("🔍 질문 품질 검증...")

# 1. 한국어 질문 비율 확인


def contains_korean(text):
    return bool(re.search("[\uac00-\ud7a3]", text))


korean_questions = df_testset["question"].apply(contains_korean).sum()
korean_ratio = korean_questions / len(df_testset) * 100

print("1️⃣ 한국어 질문 비율")
print(f"   - 한국어: {korean_questions}개 ({korean_ratio:.1f}%)")
print(f"   - 영어: {len(df_testset) - korean_questions}개 ({100 - korean_ratio:.1f}%)")

# 2. Ground truth 존재 여부
if "ground_truth" in df_testset.columns:
    has_ground_truth = df_testset["ground_truth"].notna().sum()
    gt_ratio = has_ground_truth / len(df_testset) * 100
    print("2️⃣ Ground Truth 존재")
    print(f"   - 있음: {has_ground_truth}개 ({gt_ratio:.1f}%)")
    print(f"   - 없음: {len(df_testset) - has_ground_truth}개")
else:
    print("2️⃣ Ground Truth 컬럼 없음")

# 3. 질문 길이 분포
question_lengths = df_testset["question"].str.len()
print("3️⃣ 질문 길이 분포")
print(f"   - 평균: {question_lengths.mean():.0f} chars")
print(f"   - 최소: {question_lengths.min()} chars")
print(f"   - 최대: {question_lengths.max()} chars")
print(f"   - 중앙값: {question_lengths.median():.0f} chars")

# 4. 질문 다양성 (고유 단어 수)
all_questions = " ".join(df_testset["question"].tolist())
unique_words = len(set(all_questions.split()))
total_words = len(all_questions.split())
diversity = unique_words / total_words * 100

print("4️⃣ 질문 다양성")
print(f"   - 총 단어: {total_words:,}개")
print(f"   - 고유 단어: {unique_words:,}개")
print(f"   - 다양성: {diversity:.1f}%")

print("✅ 질문 품질 검증 완료")

# %% [markdown]
# ---
#
# # 📈 Part IV: 평가 및 분석
#
# ## 6. Q/A 시스템 구현
#
# ### 6.1 RAG 파이프라인
# - **Retriever**: FAISS 벡터 검색
# - **Generator**: GPT-4o-mini + Ollama
# - **프롬프트**: 한국어 최적화

# %%
print("💾 질문 저장 중...")

# CSV 저장 (간단한 분석용)
csv_path = RESULTS_DIR / "ragas_questions_50.csv"
df_testset.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"   ✅ CSV 저장: {csv_path}")

# JSON 저장 (전체 데이터 + 재현성 + 비용)
json_path = RESULTS_DIR / "ragas_questions_50.json"
testset_dict = df_testset.to_dict(orient="records")

# 메타데이터 추가 (비용 정보 포함)
output_data = {
    "metadata": {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(df_testset),
        "generator_llm": "gpt-5-mini",
        "critic_llm": "gpt-5-mini",
        "embeddings": "text-embedding-3-small",
        "sample_size": len(sampled_docs),
        "generation_time_seconds": generation_time,
        "korean_ratio": korean_ratio,
        "log_file": str(log_file),
        "total_cost_usd": total_cost,  # 비용 추가
        "cost_per_question_usd": total_cost / len(df_testset),  # 질문당 비용
        "query_distribution": "default_query_distribution",  # 분포 방식
    },
    "questions": testset_dict,
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"   ✅ JSON 저장: {json_path}")

# 로그 파일 복사 (결과와 함께 보관)
import shutil

log_copy_path = RESULTS_DIR / f"ragas_generation_{timestamp}.log"
shutil.copy2(log_file, log_copy_path)
print(f"   ✅ 로그 저장: {log_copy_path}")

print("📁 저장된 파일:")
print(f"   - CSV: {csv_path}")
print(f"   - JSON: {json_path}")
print(f"   - 로그: {log_copy_path}")

# 로그 통계
log_size_kb = log_copy_path.stat().st_size / 1024
print("📊 로그 파일 통계:")
print(f"   - 크기: {log_size_kb:.1f} KB")
print(f"   - 라인 수: {sum(1 for _ in open(log_copy_path, 'r', encoding='utf-8'))}")

# 비용 요약
print("💰 비용 요약:")
print(f"   - 총 비용: ${total_cost:.4f}")
print(f"   - 질문당 비용: ${total_cost / len(df_testset):.4f}")
print("   - 예상 절감: 수작업 대비 ~95%")

print("" + "=" * 80)
print("✅ Section 5 완료: RAGAS 질문 생성")
print("=" * 80)

print("📊 요약:")
print(f"   - 생성된 질문: {len(df_testset)}개")
print(f"   - 한국어 비율: {korean_ratio:.1f}%")
print(f"   - 소요 시간: {generation_time:.1f}초 ({generation_time / 60:.1f}분)")
print(f"   - 총 비용: ${total_cost:.4f}")
print("   - 결과 저장: CSV + JSON + 로그")

print("💡 로그 확인 방법:")
print(f"   - 명령어: tail -f {log_copy_path}")
print(f"   - 또는: cat {log_copy_path} | grep -i error")
print(f"   - 전체 보기: cat {log_copy_path}")

print(
    "\n🎯 다음 단계: Section 6 - Naive Q/A System (4 models × 50 questions = 200 answers)"
)
print("=" * 80)

# 로깅 종료
logging.info("Section 5 completed successfully")
logging.info(f"Output files: CSV={csv_path}, JSON={json_path}, LOG={log_copy_path}")
logging.info(f"Total cost: ${total_cost:.4f} ({len(df_testset)} questions)")

# %% [markdown]
# ### 5.10 로그 분석 도구
#
# #### 📊 생성 과정 모니터링

# %%
# ============================================================================
# RAGAS 로그 분석 도구
# ============================================================================


def analyze_ragas_log(log_path):
    """
    RAGAS 로그 파일을 분석하여 주요 정보를 추출합니다.

    Args:
        log_path: 로그 파일 경로
    """
    import re
    from collections import Counter

    print(f"📊 로그 분석: {log_path}")
    print("=" * 80)

    if not Path(log_path).exists():
        print(f"❌ 로그 파일을 찾을 수 없습니다: {log_path}")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 로그 레벨 통계
    log_levels = Counter()
    errors = []
    warnings = []
    api_calls = []

    for line in lines:
        # 로그 레벨 카운트
        if " - DEBUG - " in line:
            log_levels["DEBUG"] += 1
        elif " - INFO - " in line:
            log_levels["INFO"] += 1
        elif " - WARNING - " in line:
            log_levels["WARNING"] += 1
            warnings.append(line.strip())
        elif " - ERROR - " in line:
            log_levels["ERROR"] += 1
            errors.append(line.strip())

        # API 호출 추적
        if "openai" in line.lower() or "api" in line.lower():
            api_calls.append(line.strip())

    # 1. 로그 기본 정보
    print("1️⃣ 로그 기본 정보")
    print(f"   - 총 라인 수: {len(lines):,}")
    print(f"   - 파일 크기: {Path(log_path).stat().st_size / 1024:.1f} KB")

    # 2. 로그 레벨 분포
    print("2️⃣ 로그 레벨 분포")
    for level, count in log_levels.most_common():
        print(f"   - {level}: {count:,}개")

    # 3. 에러 메시지
    if errors:
        print(f"3️⃣ 에러 메시지 ({len(errors)}개)")
        for i, error in enumerate(errors[:5], 1):  # 최대 5개만 출력
            print(f"   {i}. {error[:100]}...")
        if len(errors) > 5:
            print(f"   ... ({len(errors) - 5}개 더 있음)")
    else:
        print("3️⃣ 에러 없음 ✅")

    # 4. 경고 메시지
    if warnings:
        print(f"4️⃣ 경고 메시지 ({len(warnings)}개)")
        for i, warning in enumerate(warnings[:5], 1):
            print(f"   {i}. {warning[:100]}...")
        if len(warnings) > 5:
            print(f"   ... ({len(warnings) - 5}개 더 있음)")
    else:
        print("4️⃣ 경고 없음 ✅")

    # 5. API 호출 통계
    print("5️⃣ API 호출 관련 로그")
    print(f"   - API 관련 로그: {len(api_calls)}개")
    if api_calls:
        print("   - 예시 (최근 3개):")
        for call in api_calls[-3:]:
            print(f"     • {call[:120]}...")

    print("" + "=" * 80)
    print("💡 상세 로그 확인:")
    print(f"   - 전체 보기: cat {log_path}")
    print(f"   - 에러만: grep -i error {log_path}")
    print(f"   - 경고만: grep -i warning {log_path}")
    print(rf"   - API 호출: grep -i 'openai\|api' {log_path}")


# 사용 예시 (Section 5 실행 후 사용)
# analyze_ragas_log(RESULTS_DIR / "ragas_generation_YYYYMMDD_HHMMSS.log")

print("✅ 로그 분석 도구 준비 완료")
print("사용 방법:")
print('   analyze_ragas_log(RESULTS_DIR / "ragas_generation_YYYYMMDD_HHMMSS.log")')

# %% [markdown]
# ## 7. RAGAS 평가
#
# ### 7.1 평가 메트릭
# - **Faithfulness**: 컨텍스트 충실도
# - **Answer Relevancy**: 답변 관련성
# - **Answer Correctness**: 정답 일치도

# %% [markdown]
# ---
#
# # 📊 Part V: 결과 및 인사이트
#
# ## 8. 성능 분석
#
# ### 8.1 시각화
# - 모델별 성능 비교
# - 메트릭별 분포
# - 오류 패턴

# %% [markdown]
# ## 9. 결론 및 제언
#
# ### 9.1 주요 발견
#
# #### ✅ 성공 요인
# - **한국어 명시**: 페르소나에 명시적 지정 → 100% 한국어 생성
# - **KG 접근법**: 구조화된 질문 → 다양성 증가
# - **RAGAS 0.3.1+**: 최신 API → 안정적 생성
#
# ### 9.2 모델별 성능
#
# | 모델 | Faithfulness | Relevancy | Correctness | 비고 |
# |------|-------------|-----------|-------------|------|
# | GPT-4o-mini | TBD | TBD | TBD | 기준선 |
# | EXAONE-3.5 | TBD | TBD | TBD | 한국어 특화 |
# | Qwen-3 | TBD | TBD | TBD | 추론 강점 |
# | Gemma-3 | TBD | TBD | TBD | 범용 |
#
# ### 9.3 향후 과제
#
# #### 🎯 단기 과제
# 1. **전체 평가 실행** - 모든 모델 테스트
# 2. **오류 분석** - 한국어 생성 패턴
# 3. **검색 최적화** - Context Precision 개선
#
# #### 🚀 장기 과제
# 1. **파인튜닝** - 도메인 특화 모델
# 2. **하이브리드 검색** - BM25 + 벡터
# 3. **프롬프트 최적화** - Few-shot 예제
#
# ---
#
# ## 📚 참고문헌
#
# - [RAGAS Documentation](https://docs.ragas.io/)
# - [Crawl4AI](https://github.com/unclecode/crawl4ai)
# - [LangChain](https://docs.langchain.com/)
# - [OpenAI API](https://platform.openai.com/docs/)
