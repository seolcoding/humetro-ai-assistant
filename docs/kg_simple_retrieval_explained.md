# KG Simple Retrieval 방식 상세 설명

## 📋 개요

KG Simple은 **벡터 유사도 검색**과 **고정된 Cypher 쿼리**를 결합한 가장 안정적인 방식입니다.

---

## 🔧 핵심 구성 요소

### 1. VectorCypherRetriever 사용

```python
# Line 84-89
self.neo4j_retriever = VectorCypherRetriever(
    driver=KGRAGRetriever._driver,
    index_name="vector",              # ← 벡터 인덱스 이름
    embedder=KGRAGRetriever._embedder, # ← OpenAI embeddings
    retrieval_query=self.retrieval_query  # ← 고정된 Cypher
)
```

### 2. 고정된 Retrieval Query

```python
# Line 122-125: 단순하지만 효과적
def _build_default_retrieval_query(self) -> str:
    query = """
        // Return chunk text only (simple version)
        RETURN node.text as text, score
    """
    return query
```

---

## 🔍 동작 과정 (Step-by-Step)

### 실제 질문 예시

**질문**: "여의도역 우회에 대한 중요정보는?"

### Step 1: 질문 Embedding 생성

```python
# VectorCypherRetriever 내부에서 자동 처리
query_embedding = embedder.embed_query("여의도역 우회에 대한 중요정보는?")

# 결과: 3072차원 벡터
# [0.023, -0.145, 0.089, ..., -0.012]  # 3072개 숫자
```

### Step 2: 벡터 유사도 검색 (Neo4j)

**내부적으로 실행되는 Cypher 쿼리:**

```cypher
// VectorCypherRetriever가 자동으로 실행
CALL db.index.vector.queryNodes(
    'vector',              // 벡터 인덱스 이름
    4,                     // top_k (k=4)
    $query_embedding       // 질문의 embedding
)
YIELD node, score

// 사용자 정의 retrieval_query 실행
RETURN node.text as text, score
```

### Step 3: 결과 반환

**검색된 노드 (예시):**

```
Node 1 (score: 0.923):
text: "여의도역은 2024년 12월 14일부터 15일까지 집회로 인해
       통제되는 구간에 위치하고 있으며, 5623번, 5615번, 5618번
       노선의 우회경로가 제공됩니다..."

Node 2 (score: 0.891):
text: "우회 일시 : 12.14.(토) 00시 ~ 12.15.(일) 미정
       통제 구간 : 국회의사당 앞 전 차로
       우회방법: 서강대교 ↔ 여의나루역 ↔ 여의도역..."

Node 3 (score: 0.876):
text: "6713번 우회안내 - 통제구간: 국회 ↔ 여의지하차도(양방향)
       광흥창역,서강동주민센터 → 여의나루역 → 여의도역..."

Node 4 (score: 0.854):
text: "163번 우회안내 - 서강대교 ↔ 국회의사당 ↔ 여의도역 ↔ 대방역
       무정차 정류소: 국회의사당(19280), 국회의사당역(19133)..."
```

### Step 4: Document 변환

```python
# Line 157-179
documents = []
for item in results.items:
    content = self._format_context(item.content, item.metadata)

    doc = Document(
        page_content=content,  # ← 원본 텍스트
        metadata={
            "score": 0.923,
            "source": "knowledge_graph",
            "retrieval_type": "kg_rag"
        }
    )
    documents.append(doc)

# 최종 결과: 4개의 Document 객체
```

---

## 💡 핵심 특징

### 1. 단순함 (Simplicity)

```
질문 → Embedding → 벡터 검색 → 원본 텍스트 반환
```

**장점:**
- ✅ 이해하기 쉬움
- ✅ 디버깅 용이
- ✅ 오류 가능성 낮음

### 2. 안정성 (Stability)

**고정된 Cypher 쿼리:**
```cypher
RETURN node.text as text, score
```

- ✅ 항상 동일한 방식으로 동작
- ✅ LLM 생성 쿼리의 불확실성 없음
- ✅ 빈 결과 반환 위험 낮음

### 3. 원본 보존 (Original Text)

```python
# 그래프에 저장된 원본 청크를 그대로 반환
node.text  # ← 원본 마크다운 텍스트

# LLM 요약 없음!
# 정보 손실 없음!
```

---

## 🆚 다른 방식과 비교

### vs Naive RAG

| 측면 | Naive RAG | KG Simple |
|------|-----------|-----------|
| 검색 | FAISS (파일) | Neo4j (그래프 DB) |
| 인덱스 | 로컬 벡터 | 그래프 + 벡터 |
| 확장성 | 낮음 | 높음 (관계 추가 가능) |
| 성능 | 0.746 | 0.780 (+4.6%) |

### vs OLD Cypher

| 측면 | OLD Cypher | KG Simple |
|------|------------|-----------|
| 시작점 | ❌ 없음 (LLM 생성) | ✅ 벡터 검색 |
| 쿼리 | 동적 (불안정) | 고정 (안정) |
| 결과 | 요약 (정보 손실) | 원본 (정보 유지) |
| 성능 | 0.398 (실패) | 0.780 (성공) |

### vs FIXED Cypher

| 측면 | KG Simple | FIXED Cypher |
|------|-----------|--------------|
| 시작점 | ✅ 벡터 검색 | ✅ 벡터 검색 |
| 확장 | ❌ 없음 | ✅ 1-hop 이웃 |
| 복잡도 | 낮음 | 높음 |
| 안정성 | 높음 | 중간 |
| 성능 | **0.780** | 0.772 |

**결론**: KG Simple이 단순하면서도 가장 안정적!

---

## 📊 실제 성능 데이터

### Multi-hop Faithfulness

| Model | KG Simple | vs Naive | vs FIXED |
|-------|-----------|----------|----------|
| Gemma3-12B | **0.891** | +1.9% | +10.1% |
| GPT-4o-mini | 0.793 | +8.9% | +1.4% |
| EXAONE | 0.735 | +1.0% | +0.2% |
| GPT-OSS | 0.745 | +14.8% | +0.3% |
| Qwen3-8B | 0.739 | -1.5% | -7.9% |
| **평균** | **0.780** | **+4.6%** | **+1.1%** |

---

## 🎯 왜 KG Simple이 최고인가?

### 1. 벡터 검색의 정확성

```
Neo4j Vector Index:
- 5,879 chunks
- 3072-dimensional embeddings (text-embedding-3-large)
- Cosine similarity

→ 항상 관련된 노드를 찾음
```

### 2. 고정 Cypher의 신뢰성

```cypher
RETURN node.text as text, score

→ 복잡한 그래프 탐색 없음
→ 벡터로 찾은 노드만 반환
→ 예측 가능한 동작
```

### 3. 원본 텍스트 품질

```
Graph에 저장된 원본 청크:
- 마크다운 형식 유지
- 표, 링크, 구조 보존
- LLM 요약 없음

→ 최고 품질의 컨텍스트
```

### 4. 노이즈 최소화

```
FIXED Cypher:
벡터 노드 (4개) + 이웃 노드 (10개) = 14개
→ 일부 이웃이 관련성 낮을 수 있음 (노이즈)

KG Simple:
벡터 노드 (4개) only
→ 모두 높은 관련성 보장
```

---

## 🔬 실제 동작 예시

### 전체 Workflow

```
1. User Question:
   "여의도역 우회에 대한 중요정보는?"

2. Embedding:
   [0.023, -0.145, ..., -0.012]  (3072D)

3. Neo4j Vector Search:
   CALL db.index.vector.queryNodes('vector', 4, $embedding)

   → Found 4 nodes with scores: [0.923, 0.891, 0.876, 0.854]

4. Execute Retrieval Query:
   RETURN node.text as text, score

   → Returns original markdown text from each node

5. Format Documents:
   [
     Document(page_content="여의도역은 2024년...", score=0.923),
     Document(page_content="우회 일시: 12.14...", score=0.891),
     Document(page_content="6713번 우회안내...", score=0.876),
     Document(page_content="163번 우회안내...", score=0.854)
   ]

6. LLM Generation:
   Uses these 4 documents as context to generate answer
```

---

## 📝 코드 흐름 요약

```python
# 1. Initialization
retriever = VectorCypherRetriever(
    index_name="vector",
    retrieval_query="RETURN node.text as text, score"
)

# 2. Search (내부 동작)
def search(query_text: str, top_k: int):
    # a. Generate embedding
    embedding = embedder.embed_query(query_text)

    # b. Vector search in Neo4j
    results = neo4j.run(
        "CALL db.index.vector.queryNodes($index, $k, $emb)",
        index="vector", k=top_k, emb=embedding
    )

    # c. Execute custom Cypher on each result
    for node in results:
        text = node.text
        score = node.score
        yield RetrieverResult(content=text, metadata={"score": score})

# 3. Convert to Documents
documents = [
    Document(page_content=item.content, metadata=item.metadata)
    for item in results.items
]
```

---

## 🎓 결론

### KG Simple의 성공 요인

```
✅ 벡터 검색으로 정확한 시작점 확보
✅ 고정 Cypher로 안정적 동작 보장
✅ 원본 텍스트로 최고 품질 유지
✅ 단순함으로 오류 가능성 최소화

= 0.780 faithfulness (최고 성능)
```

### 실용적 권장

**KG Simple을 사용해야 하는 경우:**
- ✅ 안정적인 RAG 시스템이 필요할 때
- ✅ 유지보수가 중요할 때
- ✅ 대부분의 모델에서 좋은 성능이 필요할 때

**다른 방식을 고려해야 하는 경우:**
- ⚠️ Qwen3-8B 사용 시 → FIXED Cypher 고려 (0.818 > 0.739)
- ⚠️ 매우 복잡한 multi-hop 추론 → FIXED Cypher 실험

**일반적으로**: **KG Simple = Best Choice** 🏆
