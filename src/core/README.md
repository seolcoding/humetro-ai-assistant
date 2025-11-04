# Core RAG Modules

RAG 파이프라인의 핵심 재사용 모듈들입니다.

## 모듈

### `chunking.py` - 텍스트 청킹

문서를 관리 가능한 크기의 청크로 분할합니다.

**주요 기능:**
- RecursiveCharacterTextSplitter 기반
- 한국어 최적화 구분자
- 설정 가능한 청크 크기 및 오버랩
- 청크 통계 제공

**사용 예시:**
```python
from src.core.chunking import create_chunker
from langchain.schema import Document

chunker = create_chunker(chunk_size=1024, chunk_overlap=256)

# 문서 청킹
documents = [Document(page_content="텍스트 내용", metadata={"source": "file.md"})]
chunks = chunker.chunk_documents(documents)

# 통계 확인
stats = chunker.get_chunk_stats(chunks)
print(f"총 청크: {stats['total_chunks']}")
print(f"평균 길이: {stats['avg_length']:.0f} chars")
```

### `embedding.py` - 임베딩 생성 (캐시 통합)

OpenAI 임베딩 생성과 자동 캐싱을 제공합니다.

**주요 기능:**
- OpenAI text-embedding-3-small 사용
- 자동 임베딩 캐싱 (중복 API 호출 방지)
- 배치 처리 지원
- API 비용 추적

**사용 예시:**
```python
from src.core.embedding import create_embedding_generator
from src.common.logger import get_logger

logger = get_logger("my_experiment")
embedder = create_embedding_generator(
    model="text-embedding-3-small",
    logger=logger
)

# 문서 임베딩
texts = ["텍스트 1", "텍스트 2", "텍스트 3"]
embeddings = embedder.embed_documents(texts)  # 캐시 자동 확인

# 쿼리 임베딩
query_embedding = embedder.embed_query("검색 쿼리")

# 통계 확인
embedder.print_stats()
"""
============================================================
EMBEDDING STATISTICS
============================================================
API calls: 50
Cache hits: 150
Cache hit rate: 75.0%
Total cached items: 200
Total saved cost: $0.0123
============================================================
"""
```

### `vector_store.py` - FAISS 벡터 스토어

FAISS 기반 벡터 스토어 관리를 제공합니다.

**주요 기능:**
- FAISS 인덱스 생성 및 로딩
- 자동 캐싱 (디스크 저장)
- 캐시 크기 추적

**사용 예시:**
```python
from src.core.vector_store import create_vector_store_manager
from src.core.embedding import create_embedding_generator
from pathlib import Path

embedder = create_embedding_generator()
manager = create_vector_store_manager(
    embeddings=embedder,
    persist_directory=Path("./data/vectorstore")
)

# 벡터 스토어 로드 또는 생성
vectorstore = manager.get_or_create_vectorstore(documents=chunks)

# 캐시 정보
cache_info = manager.get_cache_size()
print(f"캐시 크기: {cache_info['size_mb']:.2f} MB")
```

### `knowledge_graph.py` - 지식그래프 생성

RAGAS 기반 지식그래프 생성을 제공합니다.

**주요 기능:**
- RAGAS KnowledgeGraph 생성
- Headlines, Keyphrases 추출
- 노드 및 관계 통계

**사용 예시:**
```python
from src.core.knowledge_graph import create_kg_builder

kg_builder = create_kg_builder(llm_model="gpt-4o-mini")

# 지식그래프 생성
kg = kg_builder.build_from_documents(documents)

# 통계 확인
kg_builder.print_stats(kg)
"""
============================================================
KNOWLEDGE GRAPH STATISTICS
============================================================
Total nodes: 1,234
Total relationships: 567
Node types:
  - DOCUMENT: 100
  - HEADLINE: 456
  - KEYPHRASE: 678
============================================================
"""
```

### `retrieval.py` - RAG 검색

RAG 검색 기능을 제공합니다.

**주요 기능:**
- 유사도 검색
- 스코어 기반 검색
- 검색 통계

**사용 예시:**
```python
from src.core.retrieval import create_retriever

retriever = create_retriever(vectorstore, k=5)

# 문서 검색
documents = retriever.retrieve("서울 지하철 요금은?")

# 스코어와 함께 검색
results = retriever.similarity_search_with_score("운영 시간은?")
for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content[:100]}...")

# 검색 통계
stats = retriever.get_retrieval_stats("테스트 쿼리")
print(f"검색된 문서: {stats['num_retrieved']}")
print(f"평균 스코어: {stats['avg_score']:.4f}")
```

## 완전한 파이프라인 예시

```python
from pathlib import Path
from langchain.schema import Document

from src.common.logger import get_logger
from src.core.chunking import create_chunker
from src.core.embedding import create_embedding_generator
from src.core.vector_store import create_vector_store_manager
from src.core.retrieval import create_retriever

# 1. 로거 초기화
logger = get_logger("rag_pipeline")

# 2. 문서 로드
documents = [
    Document(page_content="문서 내용 1", metadata={"source": "doc1.md"}),
    Document(page_content="문서 내용 2", metadata={"source": "doc2.md"}),
]

# 3. 청킹
chunker = create_chunker(chunk_size=1024, chunk_overlap=256)
chunks = chunker.chunk_documents(documents)
logger.info(f"Created {len(chunks)} chunks")

# 4. 임베딩 (캐시 자동)
embedder = create_embedding_generator(logger=logger)

# 5. 벡터 스토어 (캐시 자동)
vs_manager = create_vector_store_manager(
    embeddings=embedder,
    persist_directory=Path("./data/vectorstore"),
    logger=logger
)
vectorstore = vs_manager.get_or_create_vectorstore(documents=chunks)

# 6. 검색기
retriever = create_retriever(vectorstore, k=5, logger=logger)

# 7. 검색 수행
query = "서울 지하철 정보는?"
results = retriever.retrieve(query)

for i, doc in enumerate(results, 1):
    print(f"\n결과 {i}:")
    print(f"  Source: {doc.metadata.get('source', 'unknown')}")
    print(f"  Content: {doc.page_content[:100]}...")

# 8. 통계 출력
embedder.print_stats()
logger.print_summary()
```

## 테스트

```bash
# 모든 테스트 실행
pytest tests/test_core/ -v

# 개별 모듈 테스트
pytest tests/test_core/test_chunking.py -v
```

## 의존성

- `langchain` - 문서 처리 및 RAG 프레임워크
- `langchain-openai` - OpenAI 임베딩
- `langchain-community` - FAISS 벡터 스토어
- `ragas` - 지식그래프 생성
- `faiss-cpu` - 벡터 유사도 검색

## 모범 사례

### 1. 항상 로거 사용
```python
logger = get_logger("experiment_name")
# 모든 모듈에 logger 전달
```

### 2. 캐시 디렉토리 명시
```python
# 실험별로 다른 캐시 사용
embedder = create_embedding_generator(
    cache_dir=Path(f"./data/cache/exp_{experiment_id}")
)
```

### 3. 통계 확인
```python
# 각 단계 후 통계 확인
chunks = chunker.chunk_documents(documents)
stats = chunker.get_chunk_stats(chunks)
logger.info(f"Chunking stats: {stats}")
```

### 4. 에러 처리
```python
try:
    vectorstore = manager.get_or_create_vectorstore(documents=chunks)
except ValueError as e:
    logger.error(f"Vector store error: {e}")
    # Fallback 로직
```

## 성능 최적화

### 임베딩 캐싱
- 첫 실행: API 호출 (비용 발생)
- 이후 실행: 캐시 사용 (비용 0, 속도 100배 향상)

### 벡터 스토어 캐싱
- 첫 실행: FAISS 인덱스 생성 (~10초)
- 이후 실행: 디스크 로딩 (~0.5초)

### 배치 처리
```python
# 100개씩 배치 처리 (기본값)
embedder = create_embedding_generator(batch_size=100)
```
