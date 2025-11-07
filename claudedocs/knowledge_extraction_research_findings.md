# Knowledge Extraction Research Findings & Proposal

**Date:** 2025-10-29
**Purpose:** Design LLM-based structured knowledge extractor for Seoul Traffic documentation

---

## 📚 Research Summary

### 1. Existing Codebase Analysis

**Current Entity Extraction** (`src/utils/entity_preview.py`):
- **Type:** Rule-based pattern matching
- **Entities:** TRANSPORT, LOCATION, INCIDENT
- **Limitations:**
  - No relation extraction
  - Fixed keyword lists (not adaptable)
  - No semantic understanding
  - Korean-only regex patterns

**Current Schema** (`src/config/schemas.py`):
- Comprehensive Pydantic models already in place
- Well-structured metadata (PageMetadata, AttachedDocument)
- Ready for extension with knowledge graph schemas

### 2. Python Libraries for Schema.org

| Library | Purpose | Pros | Cons |
|---------|---------|------|------|
| **schemaorg** | Official Python module for schema.org | Official, well-documented | Not for advanced querying |
| **extruct** | Extract embedded metadata from HTML | Supports multiple formats (JSON-LD, microdata) | Requires pre-existing markup |
| **openschemas/extractors** | Example extractors with recipes | Ready-to-use patterns | Limited to specific domains |

**Verdict:** Schema.org is useful for **output standardization** but NOT for extraction. We need LLM-based extraction first.

### 3. Graph RAG Extraction Patterns (2024-2025)

#### Key Papers Analyzed

**iText2KG (arXiv 2409.03284)** - Most Relevant for Our Use Case
- **Approach:** Incremental, topic-independent KG construction
- **Zero-shot:** No fine-tuning needed
- **Modules:**
  1. Document Distiller (chunk text)
  2. Incremental Entity Extractor
  3. Incremental Relation Extractor
  4. Graph Integrator
- **Advantage:** No post-processing needed, handles entity deduplication

**Supply Chain KG (arXiv 2408.07705)**
- **Approach:** Zero-shot NER + Relation Extraction with LLMs
- **Accuracy:** High precision in NER and RE tasks
- **Challenge:** Public data only (no direct stakeholder info)

**Biomedical Entity/Relation (arXiv 2408.06618)**
- **Approach:** Common-knowledge-sharing mechanism
- **Pattern:** Build general knowledge graph → transfer to specific domains
- **Applicable:** Build Seoul traffic knowledge base → apply to new docs

#### Common Architecture Pattern

```
┌─────────────┐
│ Raw Text    │
│ (Markdown)  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ LLM with Pydantic   │
│ Structured Output   │
│ - Entity Extraction │
│ - Relation Extraction│
│ - Description Gen   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Entity Resolution   │
│ (Deduplication)     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Knowledge Graph     │
│ (Nodes + Edges)     │
└─────────────────────┘
```

### 4. Modern Approach: Pydantic + LLM Structured Output

**Industry Best Practice (2025):**
- OpenAI/Anthropic structured output mode
- Pydantic schema validation
- Field-level JSON output with type enforcement
- No post-processing needed

**Workflow:**
1. Define Pydantic models for entities/relations
2. Pass schema to LLM as structured output format
3. LLM returns validated JSON matching schema
4. Direct integration into knowledge graph

---

## 🏗️ Proposed Architecture

### Option A: Simple GraphRAG (Recommended for MVP)

**Tech Stack:**
- OpenAI Structured Output API
- Pydantic v2 schemas
- NetworkX for graph storage
- Schema.org types for standardization

**Process:**
```python
# 1. Define Pydantic schemas
class Entity(BaseModel):
    id: str
    type: Literal["TransportLine", "Location", "Incident", "Organization", "Policy"]
    name: str
    description: str
    properties: Dict[str, Any]

class Relation(BaseModel):
    source_id: str
    target_id: str
    relation_type: str
    description: str
    confidence: float

class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relations: List[Relation]
    metadata: Dict[str, Any]

# 2. Extract with LLM structured output
kg = llm.extract_knowledge_graph(
    markdown_text=md_content,
    response_format=KnowledgeGraph
)

# 3. Store in graph database
```

**Advantages:**
- Fast implementation (2-3 days)
- No training needed
- Works with existing OpenAI API
- Easy to iterate and refine

**Disadvantages:**
- API costs for large-scale processing
- Limited to LLM context window
- May miss domain-specific nuances

### Option B: iText2KG-Inspired (Production-Grade)

**Tech Stack:**
- iText2KG methodology
- Custom entity resolution
- Incremental graph construction
- Fine-tuned local LLM (optional)

**Process:**
```python
# 1. Document Distiller
chunks = distill_document(markdown, chunk_size=1000)

# 2. Incremental Entity Extraction
entities = []
for chunk in chunks:
    new_entities = extract_entities(chunk)
    entities = merge_entities(entities, new_entities)  # Deduplication

# 3. Incremental Relation Extraction
relations = []
for entity_pair in combinations(entities, 2):
    if context_supports_relation(entity_pair, chunks):
        relation = extract_relation(entity_pair, chunks)
        relations.append(relation)

# 4. Graph Integration
kg = build_graph(entities, relations)
```

**Advantages:**
- Better entity deduplication
- Handles large documents (no context limit)
- More accurate for complex domains
- Lower API costs (fewer LLM calls)

**Disadvantages:**
- More complex implementation (1-2 weeks)
- Requires careful tuning
- More code to maintain

---

## 📋 Entity & Relation Schema Design

### Seoul Traffic Domain Entities

Based on schema.org + domain-specific types:

```python
class TransportEntity(str, Enum):
    """Transportation entities (schema.org/TransportAction)"""
    SUBWAY_LINE = "subway_line"          # 지하철 노선
    BUS_ROUTE = "bus_route"              # 버스 노선
    STATION = "station"                  # 역
    TERMINAL = "terminal"                # 터미널
    INTERCHANGE = "interchange"          # 환승역
    TRANSPORT_SERVICE = "transport_service"  # 교통 서비스

class LocationEntity(str, Enum):
    """Location entities (schema.org/Place)"""
    DISTRICT = "district"                # 구
    NEIGHBORHOOD = "neighborhood"        # 동
    ROAD = "road"                        # 도로
    INTERSECTION = "intersection"        # 교차로
    LANDMARK = "landmark"                # 랜드마크

class IncidentEntity(str, Enum):
    """Incident entities (custom)"""
    DELAY = "delay"                      # 지연
    ACCIDENT = "accident"                # 사고
    CONSTRUCTION = "construction"        # 공사
    CLOSURE = "closure"                  # 폐쇄
    DETOUR = "detour"                    # 우회

class PolicyEntity(str, Enum):
    """Policy entities (schema.org/GovernmentService)"""
    POLICY = "policy"                    # 정책
    REGULATION = "regulation"            # 규제
    SUBSIDY = "subsidy"                  # 보조금
    SERVICE = "service"                  # 서비스

class TemporalEntity(str, Enum):
    """Temporal entities (schema.org/DateTime)"""
    DATE = "date"                        # 날짜
    TIME_PERIOD = "time_period"          # 기간
    SCHEDULE = "schedule"                # 스케줄
```

### Relation Types

```python
class RelationType(str, Enum):
    """Relation types between entities"""

    # Structural Relations
    PART_OF = "part_of"                  # A는 B의 일부
    CONNECTS_TO = "connects_to"          # A는 B와 연결
    LOCATED_IN = "located_in"            # A는 B에 위치
    OPERATES_IN = "operates_in"          # A는 B에서 운영

    # Temporal Relations
    OCCURS_ON = "occurs_on"              # A는 B에 발생
    DURING = "during"                    # A는 B 기간 중
    BEFORE = "before"                    # A는 B 이전
    AFTER = "after"                      # A는 B 이후

    # Causal Relations
    CAUSES = "causes"                    # A는 B를 유발
    AFFECTS = "affects"                  # A는 B에 영향
    RESULTS_IN = "results_in"            # A는 B로 귀결

    # Policy Relations
    IMPLEMENTS = "implements"            # A는 B를 시행
    REGULATES = "regulates"              # A는 B를 규제
    PROVIDES = "provides"                # A는 B를 제공
```

### Complete Pydantic Schemas

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from datetime import datetime

class Entity(BaseModel):
    """Knowledge graph entity"""
    id: str = Field(..., description="Unique entity identifier")
    type: str = Field(..., description="Entity type (TransportEntity, LocationEntity, etc.)")
    name: str = Field(..., description="Entity name in Korean")
    name_en: Optional[str] = Field(None, description="Entity name in English (if available)")
    description: str = Field(..., description="Brief description of entity")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    schema_org_type: Optional[str] = Field(None, description="Schema.org type for standardization")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence (0-1)")
    source_url: str = Field(..., description="Source document URL")
    source_context: str = Field(..., description="Text snippet where entity was mentioned")

class Relation(BaseModel):
    """Knowledge graph relation"""
    id: str = Field(..., description="Unique relation identifier")
    source_entity_id: str = Field(..., description="Source entity ID")
    target_entity_id: str = Field(..., description="Target entity ID")
    relation_type: RelationType = Field(..., description="Type of relation")
    description: str = Field(..., description="Human-readable relation description")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence (0-1)")
    source_url: str = Field(..., description="Source document URL")
    source_context: str = Field(..., description="Text snippet supporting this relation")
    temporal_scope: Optional[str] = Field(None, description="Temporal validity of relation")

class KnowledgeGraph(BaseModel):
    """Complete knowledge graph from document"""
    document_url: str
    document_title: str
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    entities: List[Entity] = Field(..., description="Extracted entities")
    relations: List[Relation] = Field(..., description="Extracted relations")
    summary: str = Field(..., description="Document summary for context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
```

---

## 🚀 Implementation Roadmap

### Phase 1: MVP (Week 1)

**Goal:** Basic entity + relation extraction from markdown files

**Tasks:**
1. ✅ Research and design (DONE)
2. Define Pydantic schemas for entities/relations
3. Implement OpenAI structured output extractor
4. Test on 10 sample documents
5. Validate JSON output structure

**Deliverables:**
- `src/knowledge/schemas.py` - Pydantic models
- `src/knowledge/extractor.py` - LLM-based extractor
- `tests/test_knowledge_extraction.py` - Unit tests
- Sample output JSON files

### Phase 2: Graph Storage (Week 2)

**Goal:** Store extracted knowledge in graph database

**Tasks:**
1. Choose graph storage (NetworkX vs Neo4j vs TigerGraph)
2. Implement graph builder from extracted entities/relations
3. Add entity deduplication logic
4. Create visualization utilities
5. Build query interface

**Deliverables:**
- `src/knowledge/graph_builder.py` - Graph construction
- `src/knowledge/graph_store.py` - Persistence layer
- `src/knowledge/visualizer.py` - Graph visualization
- Interactive graph explorer

### Phase 3: Production Optimization (Week 3-4)

**Goal:** Scale to full dataset with quality assurance

**Tasks:**
1. Implement batch processing pipeline
2. Add confidence thresholding
3. Create human-in-the-loop validation
4. Optimize API costs (caching, batch processing)
5. Add incremental updates
6. Performance testing

**Deliverables:**
- `src/scripts/extract_knowledge_batch.py` - Batch extraction
- Quality metrics dashboard
- Cost analysis report
- Documentation

---

## 💡 Key Decisions Needed

### 1. LLM Provider
- **OpenAI GPT-4o:** Best structured output, higher cost
- **Anthropic Claude 3.5:** Good reasoning, cheaper
- **Local Llama 3:** Zero cost, requires GPU, lower quality

### 2. Graph Storage
- **NetworkX (In-Memory):** Fast prototyping, limited scale
- **Neo4j (Graph DB):** Production-grade, requires server
- **JSON Files:** Simplest, no querying capabilities

### 3. Processing Approach
- **Real-time:** Extract during crawling (slower crawls)
- **Batch:** Process after crawling (better separation of concerns)
- **Hybrid:** Extract metadata during crawl, deep extraction in batch

### 4. Quality Control
- **Confidence Thresholds:** Reject low-confidence extractions
- **Human Validation:** Sample review (5-10% of extractions)
- **Cross-Document Validation:** Verify entities across multiple sources

---

## 📊 Expected Outcomes

### Metrics
- **Entity Count:** 500-1,000 unique entities per 1,000 documents
- **Relation Count:** 2,000-5,000 relations per 1,000 documents
- **Accuracy:** 85-90% precision (with confidence > 0.7)
- **Coverage:** 70-80% of key facts captured

### Use Cases
1. **FAQ Generation:** Auto-generate Q&A from knowledge graph
2. **Semantic Search:** Find related content via graph traversal
3. **Timeline Construction:** Track policy changes over time
4. **Impact Analysis:** Understand cascading effects of incidents
5. **Recommendation:** Suggest related transportation services

---

## 📝 Next Steps

1. **Discuss & Decide:**
   - LLM provider choice
   - Graph storage approach
   - Processing strategy

2. **Prototype (3-5 days):**
   - Implement Option A (Simple GraphRAG)
   - Test on 10 sample documents
   - Validate output quality

3. **Iterate:**
   - Review extraction quality
   - Refine prompts and schemas
   - Add missing entity/relation types

4. **Scale:**
   - Process full dataset (59+ documents)
   - Build graph database
   - Create query interface

---

**References:**
- iText2KG Paper: arXiv:2409.03284
- GraphRAG Survey: arXiv:2501.00309
- Supply Chain KG: arXiv:2408.07705
- Schema.org: https://schema.org/
- OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
