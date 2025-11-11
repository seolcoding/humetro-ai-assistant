# Knowledge Graph RAG Architecture Comparison
**우리 실험의 Naive KG vs Cypher KG 아키텍처 분석**

## Executive Summary

우리의 실험은 아티클의 "Text-to-Cypher" 방식이 **아닙니다**. 대신 **하이브리드 패턴 1 (Vector → Graph)**의 두 가지 변형을 비교합니다:

1. **Naive KG**: Vector only (그래프 관계 미활용)
2. **Cypher KG**: Vector + Graph Expansion (하이브리드 완전 구현)

---

## 아티클 프레임워크 매핑

### 우리 실험 vs 아티클의 2가지 패러다임

| 아티클 방법론 | 우리 실험 대응 | 사용 여부 |
|--------------|--------------|---------|
| **Text-to-Cypher** (LLM이 동적 쿼리 생성) | ❌ 사용 안 함 | N/A |
| **Community Detection** (비정형→그래프 생성→커뮤니티 요약) | ❌ 사용 안 함 | N/A |
| **Hybrid Pattern 1** (Vector → Graph Enrichment) | ✅ **우리 실험의 기반** | Both |

---

## 1. Naive KG (Simple KG RAG) - `kg_rag_retriever.py`

### 아키텍처

```
사용자 질의
    ↓
[Vector Search]  ← Neo4j vector index로 top-k 노드 찾기
    ↓
[Fixed Cypher]   ← 고정 쿼리: RETURN node.text, score
    ↓
[Return Chunks]  ← 원본 텍스트만 반환
```

### 구현 코드 (Line 84-126)

```python
# 1. Neo4j GraphRAG 라이브러리 사용
self.neo4j_retriever = VectorCypherRetriever(
    driver=driver,
    index_name="vector",          # Vector similarity index
    embedder=OpenAIEmbeddings(),  # text-embedding-3-large
    retrieval_query="""
        RETURN node.text as text, score
    """
)

# 2. 검색 실행
results = self.neo4j_retriever.search(
    query_text=query,
    top_k=self.k  # 기본값: 4
)

# 3. LangChain Document로 변환
documents = [
    Document(
        page_content=item.content,
        metadata={"source": "knowledge_graph", "retrieval_type": "kg_rag"}
    )
    for item in results.items
]
```

### 아티클 개념 매핑

| 아티클 단계 | Naive KG 구현 | 상태 |
|-----------|--------------|------|
| (1) **Recall (Vector)** | `db.index.vector.queryNodes()` | ✅ 있음 |
| (2) **Entry Point** | Vector로 찾은 top-k 노드 | ✅ 있음 |
| (3) **Graph Traversal** | 없음 (그래프 관계 탐색 안 함) | ❌ **없음** |
| (4) **Enrichment** | 없음 (이웃 노드 확장 안 함) | ❌ **없음** |
| (5) **Context Aggregation** | Vector 결과만 반환 | ⚠️ 단순 |

### 왜 "Naive"인가?

**"Naive"의 의미:**
- 그래프의 **관계(Edges)**를 전혀 활용하지 않음
- 단순히 Neo4j를 "Vector Database"로만 사용
- 일반 FAISS Vector Store와 본질적으로 동일한 동작

**아티클 관점:**
- "하이브리드 패턴 1"의 **Step 1만 구현**한 버전
- Graph Enrichment 없음 → "Precision" 향상 효과 없음

---

## 2. Cypher KG (Fixed Hybrid) - `kg_cypher_retriever.py`

### 아키텍처

```
사용자 질의
    ↓
[Vector Search]        ← Neo4j vector index로 top-k starting nodes 찾기
    ↓
[Graph Expansion]      ← Starting nodes의 1-hop/2-hop neighbors 탐색
    ↓
[Context Aggregation]  ← Starting + Neighbors 텍스트 수집
    ↓
[Return Enriched]      ← 확장된 컨텍스트 반환
```

### 구현 코드 (Line 130-228)

**Step 1: Vector Search (Starting Point)**
```python
# Line 133-158
query_embedding = embedder.embed_query(query)

vector_results = session.run("""
    CALL db.index.vector.queryNodes('vector', $k, $embedding)
    YIELD node, score
    RETURN elementId(node) as nodeId, node.text as text, score
    ORDER BY score DESC
""", k=self.k, embedding=query_embedding)

starting_nodes = [
    {'id': record['nodeId'], 'text': record['text'], 'score': record['score']}
    for record in vector_results
]
```

**Step 2: Graph Expansion (1-hop Neighbors)**
```python
# Line 164-178
expansion_query = """
    UNWIND $nodeIds as startNodeId
    CALL {
        WITH startNodeId
        MATCH (start)
        WHERE elementId(start) = startNodeId
        OPTIONAL MATCH (start)-[r]-(neighbor)  ← 관계 탐색!
        WHERE neighbor.text IS NOT NULL
        RETURN DISTINCT neighbor.text as text
    }
    RETURN DISTINCT text
"""

neighbor_texts = session.run(expansion_query, nodeIds=node_ids)
```

**Step 3: Context Aggregation**
```python
# Line 201-207
all_texts = [starting_nodes] + [neighbors]
all_texts = list(dict.fromkeys(all_texts))  # 중복 제거

documents = [
    Document(
        page_content=text,
        metadata={
            "is_starting_node": (i < len(starting_nodes)),
            "expansion_hops": self.expansion_hops
        }
    )
    for i, text in enumerate(all_texts)
]
```

### 아티클 개념 매핑

| 아티클 단계 | Cypher KG 구현 | 상태 |
|-----------|---------------|------|
| (1) **Recall (Vector)** | `db.index.vector.queryNodes()` | ✅ 완전 구현 |
| (2) **Entry Point** | Vector로 찾은 starting nodes | ✅ 완전 구현 |
| (3) **Graph Traversal** | `MATCH (start)-[r]-(neighbor)` | ✅ **완전 구현** |
| (4) **Enrichment** | Starting + Neighbors 결합 | ✅ **완전 구현** |
| (5) **Context Aggregation** | 중복 제거 + 순서 보존 | ✅ 완전 구현 |

### 왜 "Cypher"인가?

**원래 의도 (FAILED):**
```python
# OLD (Line 9): Pure Cypher generation without vector search
# LLM이 동적으로 Cypher 쿼리 생성 → 실패 (-47% 성능 저하)
```

**수정된 구현 (FIXED):**
```python
# NEW (Line 10): Vector search first → then graph traversal
# 고정된 프로그래밍 방식 Cypher → 성공 (+108.5% 향상)
```

**"Cypher"의 의미:**
- 이름은 유지했지만, 실제로는 **LLM Text-to-Cypher가 아님**
- **프로그래밍 방식의 고정 Cypher 쿼리** 사용
- 아티클의 "하이브리드 패턴 1" 완전 구현

---

## 🔬 핵심 차이점 비교표

| 구분 | Naive KG | Cypher KG (Fixed) | 아티클 개념 |
|------|---------|------------------|-----------|
| **Vector Search** | ✅ Top-k nodes | ✅ Top-k starting nodes | Recall |
| **Graph Traversal** | ❌ **없음** | ✅ **1-hop/2-hop** | Enrichment |
| **반환 데이터** | Starting nodes만 | Starting + Neighbors | Aggregation |
| **컨텍스트 확장** | 없음 | 관계로 연결된 추가 청크 | Precision |
| **아티클 패턴** | 패턴 1 - Step 1만 | **패턴 1 - 완전 구현** | Hybrid |
| **구현 방식** | Neo4j GraphRAG 라이브러리 | 직접 Cypher 프로그래밍 | N/A |

---

## 🎓 아티클 프레임워크로 설명하면

### Naive KG (Simple)
```
"하이브리드 패턴 1의 불완전한 구현"

✅ Vector Recall 단계만 사용
❌ Graph Precision/Enrichment 단계 없음
→ 결과: 일반 Vector RAG와 거의 동일한 성능
```

### Cypher KG (Fixed)
```
"하이브리드 패턴 1의 완전한 구현"

✅ Vector Recall (starting point)
✅ Graph Entry Point (벡터로 찾은 노드)
✅ Graph Enrichment (관계 탐색으로 neighbor 추가)
✅ Context Aggregation (starting + neighbors)
→ 결과: +11% vs Naive RAG, +6% vs Naive KG
```

---

## 📈 성능 차이의 이유 (아티클 관점)

### Naive KG가 Naive RAG과 비슷한 이유
```
Naive KG = "Neo4j에 저장된 Vector RAG"
- 그래프 관계를 전혀 활용 안 함
- 단순 벡터 유사도만 사용
→ 아티클 표현: "Vector RAG with extra steps"
```

### Cypher KG가 더 나은 이유 (아티클 4.2절)
```
Cypher KG = "하이브리드 패턴 1 완전 구현"
- Vector로 진입점 확보 (Recall)
- Graph로 관계 확장 (Enrichment)
→ 아티클 표현: "Vector RAG의 속도 + Graph RAG의 깊이"
```

**아티클의 예시 (72번 인용):**
> "Vector 검색이 '가죽 소파'에 대한 문서를 찾았을 때, 그래프 탐색을 통해 '(가죽 소파)-[재질: 가죽]'이라는 관계를 확인합니다."

**우리 예시:**
```
질의: "여객선 소요시간"
Vector Search → "여객선_이용안내" 문서 찾음
Graph Expansion → 관계로 연결된 "여객선_운항정보", "요금_안내" 추가 발견
Result → 더 풍부한 컨텍스트로 정확한 답변 생성
```

---

## 🚨 중요: 우리는 Text-to-Cypher 방식이 아닙니다!

### 아티클의 Text-to-Cypher (우리가 사용하지 않는 방식)
```python
# 아티클 방법 1: LLM이 동적으로 Cypher 생성
user_query = "2025년 3분기 기술 섹터 딜은?"
llm_generated_cypher = llm.generate(
    f"Convert to Cypher: {user_query}"
)
# Output: "MATCH (d:Deal)-[:IN_SECTOR]->(:Sector {name: 'Tech'})
#          WHERE d.date > '2025-07-01' RETURN d"
```

### 우리의 방식: 고정된 프로그래밍 Cypher
```python
# 우리 방식: 미리 작성된 고정 쿼리
expansion_query = """
    MATCH (start)-[r]-(neighbor)
    WHERE neighbor.text IS NOT NULL
    RETURN neighbor.text
"""
# 모든 쿼리에 대해 동일한 패턴 사용
```

**차이점:**
- 아티클 방법 1: **LLM이 쿼리 생성** → 신뢰성 문제, 파인튜닝 필요
- 우리 방법: **프로그래머가 쿼리 작성** → 신뢰성 높음, 유연성 낮음

---

## 📊 성능 결과 (CHECKPOINT 문서 기반)

### Quick Test (10Q, 3M)

| Method | Faithfulness | vs Naive RAG | vs Naive KG |
|--------|-------------|--------------|------------|
| **Naive RAG** | 0.746 | baseline | -4.6% |
| **Naive KG** | 0.780 | +4.6% | baseline |
| **Cypher KG (Fixed)** | **0.830** | **+11.2%** 🏆 | **+6.4%** 🥇 |

### 아티클 프레임워크로 해석

**Naive KG +4.6% 향상:**
- 아티클: "Vector RAG와 거의 동일, 약간의 인덱스 최적화 효과"
- 실제: Neo4j vector index가 FAISS보다 약간 나음

**Cypher KG +11.2% 향상:**
- 아티클 (17번 인용): "Vector의 속도 + Graph의 정확성/컨텍스트"
- 실제: Graph Enrichment가 multi-hop 질문에 효과적

---

## 🎯 우리 실험의 핵심 기여

### 아티클이 제시한 이론적 프레임워크
```
"하이브리드 패턴 1: Vector (Recall) → Graph (Precision/Enrichment)"
- 개념: 아티클 17번 인용
- 구현 예시: LlamaIndex PropertyGraphIndex
```

### 우리 실험의 실증적 검증
```
"하이브리드 패턴 1"을 Naive RAG, Naive KG, Full Hybrid로 분해하여
각 단계의 성능 기여도를 정량적으로 측정

→ Graph Enrichment의 순수 효과: +6.4% (Cypher vs Naive KG)
→ Vector + Graph 통합 효과: +11.2% (Cypher vs Naive RAG)
```

---

## 🔍 우리 실험이 아티클의 어떤 부분을 검증했는가?

### 아티클 주장 (17번 인용)
> "가장 균형 잡힌 방법: Vector RAG의 속도 + Graph RAG의 정확성/컨텍스트"

### 우리 실험의 검증
✅ **검증됨**: Cypher KG (Hybrid) > Naive KG (Vector only)
✅ **정량화**: +6.4% faithfulness 향상 (graph enrichment 효과)
✅ **실용성**: 고정 Cypher로 Text-to-Cypher의 신뢰성 문제 회피

### 아티클이 다루지 않은 우리의 발견
```
⚠️ "Pure Text-to-Cypher without vector search = -47% 성능 저하"

아티클은 Text-to-Cypher의 "신뢰성 문제"만 언급했지만,
우리는 "starting point 부재" 문제를 실증적으로 발견했습니다.

→ 시사점: Vector search는 선택이 아닌 필수
```

---

## 📚 구현 파일 참조

| 파일 | 라인 | 설명 |
|------|------|------|
| `src/kg_agent/kg_rag_retriever.py` | 84-126 | Naive KG: VectorCypherRetriever 사용 |
| `src/kg_agent/kg_cypher_retriever.py` | 8-17 | Critical Fix 주석 (OLD vs NEW) |
| `src/kg_agent/kg_cypher_retriever.py` | 133-158 | Vector search implementation |
| `src/kg_agent/kg_cypher_retriever.py` | 164-199 | Graph expansion (1-hop/2-hop) |
| `src/kg_agent/kg_cypher_retriever.py` | 201-228 | Context aggregation |

---

## 💡 결론: 우리 실험의 위치

### 아티클의 2×2 매트릭스

|  | 정형 KG 존재 | 비정형 텍스트 |
|--|------------|-------------|
| **동적 쿼리 (LLM)** | Text-to-Cypher (방법 1) | N/A |
| **고정 쿼리 (Programmatic)** | **우리 실험** ✅ | Community Detection (방법 2) |

### 우리의 선택

**데이터:** 정형 KG (Neo4j에 구축됨)
**쿼리 방식:** 고정된 프로그래밍 Cypher
**검색 전략:** 하이브리드 (Vector → Graph)

**아티클 관점:**
```
우리는 "Text-to-Cypher의 신뢰성 문제"를 회피하면서
"하이브리드 패턴 1의 Enrichment 효과"를 얻는
실용적인 중간 지점을 선택했습니다.

Trade-off:
- 포기: Text-to-Cypher의 "질의 유연성" (아티클 1.2절)
- 획득: "높은 신뢰성" + "Graph Enrichment" (아티클 4.2절)
```

---

## 🚀 미래 개선 방향 (아티클 기반)

### 현재 위치
```
[단순] Naive KG ──────→ [우리] Cypher KG ──────→ [고급] ???
         (Vector only)        (Vector+Graph-1hop)
```

### 아티클이 제시한 Next Steps

1. **2-hop Expansion** (이미 지원, `expansion_hops=2`)
2. **Dynamic Community Selection** (아티클 51번 - 쿼리에 따라 탐색 깊이 조정)
3. **GNN Integration** (아티클 68번 - NVIDIA G-Retriever 방식)
4. **Query Router** (아티클 30번 - 단순 쿼리는 Naive, 복잡한 쿼리는 Cypher)

### 실험에 추가 가능한 변형

```python
# Option 3: Dynamic Depth (아티클 권장)
if query_complexity == "simple":
    expansion_hops = 0  # Naive KG와 동일
elif query_complexity == "medium":
    expansion_hops = 1  # 현재 Cypher KG
else:
    expansion_hops = 2  # 고급 탐색
```

---

**요약:** 우리의 Naive KG vs Cypher KG 실험은 아티클의 "하이브리드 패턴 1"을 **단계적으로 구현**하여 각 단계의 기여도를 실증적으로 측정한 연구입니다.
