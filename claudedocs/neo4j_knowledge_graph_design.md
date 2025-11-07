# Neo4j Knowledge Graph Design for Seoul Traffic Documentation

**Date:** 2025-10-29
**Updated:** Neo4j as graph database choice
**Purpose:** Production-grade knowledge graph architecture with Neo4j

---

## 🏗️ Architecture Overview

```
┌──────────────────┐
│ Markdown Files   │
│ (59+ docs)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│ LLM Entity/Relation      │
│ Extractor                │
│ (OpenAI Structured Out)  │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Pydantic Validation      │
│ + Entity Resolution      │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Neo4j Graph Database     │
│ - Nodes (Entities)       │
│ - Relationships          │
│ - Properties             │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Query Interface          │
│ - Cypher Queries         │
│ - Vector Search          │
│ - GraphRAG Retrieval     │
└──────────────────────────┘
```

---

## 📊 Neo4j Schema Design

### Node Labels (Entity Types)

Based on Seoul Traffic domain + schema.org types:

#### 1. Transport Nodes

```cypher
// Subway Line
(:SubwayLine {
    id: "line_2",
    name: "2호선",
    name_en: "Line 2",
    line_number: 2,
    color: "#009D3E",
    total_stations: 43,
    description: "서울 지하철 2호선",
    schema_org_type: "TransitLine",
    source_url: "https://news.seoul.go.kr/...",
    created_at: datetime(),
    updated_at: datetime()
})

// Bus Route
(:BusRoute {
    id: "bus_153",
    name: "153번",
    route_type: "간선",
    color: "#0068B7",
    description: "강남역-잠실 간선버스",
    schema_org_type: "BusRoute"
})

// Station
(:Station {
    id: "gangnam",
    name: "강남역",
    name_en: "Gangnam Station",
    latitude: 37.498095,
    longitude: 127.027610,
    address: "서울시 강남구...",
    is_interchange: true,
    schema_org_type: "BusStation"
})

// Terminal
(:Terminal {
    id: "express_terminal",
    name: "고속터미널",
    terminal_type: "bus_terminal",
    description: "서울고속버스터미널"
})
```

#### 2. Location Nodes

```cypher
// District (구)
(:District {
    id: "gangnam_gu",
    name: "강남구",
    name_en: "Gangnam-gu",
    population: 561052,
    area_km2: 39.5,
    schema_org_type: "AdministrativeArea"
})

// Neighborhood (동)
(:Neighborhood {
    id: "yeoksam_dong",
    name: "역삼동",
    postal_code: "06234"
})

// Road
(:Road {
    id: "teheran_ro",
    name: "테헤란로",
    name_en: "Teheran-ro",
    length_km: 3.7,
    lanes: 8
})

// Intersection
(:Intersection {
    id: "gangnam_intersection",
    name: "강남역교차로",
    latitude: 37.498,
    longitude: 127.028
})
```

#### 3. Incident Nodes

```cypher
// Delay
(:Delay {
    id: "delay_20250129_001",
    incident_type: "signal_failure",
    severity: "medium",
    start_time: datetime("2025-01-29T08:30:00"),
    end_time: datetime("2025-01-29T09:15:00"),
    affected_lines: ["2호선"],
    description: "신호 장애로 인한 지연",
    resolved: true
})

// Accident
(:Accident {
    id: "accident_20250128_002",
    accident_type: "collision",
    severity: "high",
    casualties: 0,
    description: "차량 충돌 사고"
})

// Construction
(:Construction {
    id: "construction_gangnam_2025",
    project_name: "강남역 환승센터 공사",
    start_date: date("2025-01-01"),
    end_date: date("2025-12-31"),
    budget: 15000000000,
    status: "in_progress"
})
```

#### 4. Policy Nodes

```cypher
// Policy
(:Policy {
    id: "climate_card_2024",
    name: "기후동행카드",
    name_en: "Climate Card",
    policy_type: "transportation_subsidy",
    launch_date: date("2024-01-27"),
    monthly_fee: 65000,
    description: "대중교통 무제한 이용 정기권",
    status: "active",
    schema_org_type: "GovernmentService"
})

// Service
(:Service {
    id: "han_river_bus",
    name: "한강버스",
    service_type: "public_transport",
    operator: "서울시",
    launch_date: date("2024-09-27")
})

// Organization
(:Organization {
    id: "seoul_metro",
    name: "서울교통공사",
    name_en: "Seoul Metro",
    org_type: "public_corporation",
    schema_org_type: "Organization"
})
```

#### 5. Temporal Nodes

```cypher
// Date (for aggregation)
(:Date {
    date: date("2025-01-29"),
    year: 2025,
    month: 1,
    day: 29,
    day_of_week: "Wednesday",
    is_holiday: false
})

// Schedule
(:Schedule {
    id: "line2_weekday_schedule",
    schedule_type: "weekday",
    first_train: time("05:30:00"),
    last_train: time("00:30:00"),
    interval_peak: 3,
    interval_off_peak: 6
})
```

#### 6. Document Nodes (for RAG)

```cypher
// Document (Source)
(:Document {
    id: "doc_traffic_archives_515614",
    url: "https://news.seoul.go.kr/traffic/archives/515614",
    title: "11월 1일, 다시 달리는 한강버스...",
    published_date: date("2024-10-28"),
    word_count: 536,
    embedding: [0.123, 0.456, ...],  // Vector embedding
    summary: "한강버스 재운항 관련 안내"
})

// Chunk (for fine-grained RAG)
(:Chunk {
    id: "chunk_515614_001",
    text: "서울시는 11월 1일부터...",
    chunk_index: 0,
    embedding: [0.234, 0.567, ...]
})
```

### Relationship Types

#### 1. Structural Relationships

```cypher
// Transport Structure
(:SubwayLine)-[:INCLUDES_STATION]->(:Station)
(:BusRoute)-[:STOPS_AT]->(:Station)
(:Station)-[:CONNECTS_TO]->(:Station)
(:Station)-[:INTERCHANGE_WITH]->(:SubwayLine)

// Location Hierarchy
(:Neighborhood)-[:PART_OF]->(:District)
(:District)-[:PART_OF]->(:City)
(:Station)-[:LOCATED_IN]->(:District)
(:Road)-[:LOCATED_IN]->(:District)
```

#### 2. Operational Relationships

```cypher
// Service Operations
(:Organization)-[:OPERATES]->(:SubwayLine)
(:Organization)-[:MANAGES]->(:Station)
(:Service)-[:PROVIDED_BY]->(:Organization)

// Transport Connections
(:SubwayLine)-[:INTERSECTS]->(:SubwayLine)
(:BusRoute)-[:CONNECTS]->(:Terminal)
```

#### 3. Incident Relationships

```cypher
// Incident Impact
(:Delay)-[:AFFECTS]->(:SubwayLine)
(:Accident)-[:OCCURRED_AT]->(:Station)
(:Construction)-[:BLOCKS]->(:Road)
(:Delay)-[:CAUSES]->(:Delay)  // Cascading effects

// Temporal
(:Delay)-[:OCCURRED_ON]->(:Date)
```

#### 4. Policy Relationships

```cypher
// Policy Implementation
(:Organization)-[:IMPLEMENTS]->(:Policy)
(:Policy)-[:APPLIES_TO]->(:SubwayLine)
(:Policy)-[:APPLIES_TO]->(:BusRoute)
(:Service)-[:FUNDED_BY]->(:Policy)
```

#### 5. Document Relationships (for RAG)

```cypher
// Document References
(:Document)-[:MENTIONS]->(:Entity)  // Generic entity reference
(:Document)-[:ABOUT]->(:Policy)
(:Document)-[:DESCRIBES]->(:Construction)
(:Document)-[:HAS_CHUNK]->(:Chunk)

// Entity Extraction
(:Chunk)-[:CONTAINS_ENTITY]->(:Entity)
(:Entity)-[:EXTRACTED_FROM]->(:Document)
```

---

## 🔧 Implementation Stack

### Required Dependencies

```toml
# pyproject.toml
dependencies = [
    "neo4j>=5.26.0",              # Official Neo4j driver
    "neo4j-graphrag-python>=0.1.0",  # Neo4j GraphRAG package (2025)
    "openai>=1.59.0",             # For LLM extraction
    "pydantic>=2.10.0",           # Schema validation
    "langchain>=0.3.0",           # Optional: LangChain integration
    "langchain-neo4j>=0.3.0",     # Optional: Neo4j LangChain tools
]
```

### Connection Setup

```python
from neo4j import GraphDatabase
from neo4j_graphrag.llm_graph_builder import LLMEntityRelationExtractor
from typing import Optional

class Neo4jKnowledgeGraph:
    """Neo4j Knowledge Graph Manager"""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database

    def close(self):
        self.driver.close()

    def create_indexes(self):
        """Create indexes for faster querying"""
        with self.driver.session(database=self.database) as session:
            # Node indexes
            session.run("CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)")
            session.run("CREATE INDEX station_name IF NOT EXISTS FOR (s:Station) ON (s.name)")
            session.run("CREATE INDEX document_url IF NOT EXISTS FOR (d:Document) ON (d.url)")

            # Full-text search indexes
            session.run("""
                CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
                FOR (e:Entity|Station|SubwayLine|BusRoute)
                ON EACH [e.name, e.description]
            """)

            # Vector index for embeddings (RAG)
            session.run("""
                CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
                FOR (d:Document) ON (d.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }}
            """)

    def create_constraints(self):
        """Create uniqueness constraints"""
        with self.driver.session(database=self.database) as session:
            # Ensure unique IDs
            session.run("CREATE CONSTRAINT entity_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            session.run("CREATE CONSTRAINT document_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.url IS UNIQUE")
```

---

## 📝 Pydantic Models for Neo4j

### Entity Models

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from datetime import datetime
from enum import Enum

class EntityType(str, Enum):
    """Entity types matching Neo4j node labels"""
    SUBWAY_LINE = "SubwayLine"
    BUS_ROUTE = "BusRoute"
    STATION = "Station"
    TERMINAL = "Terminal"
    DISTRICT = "District"
    NEIGHBORHOOD = "Neighborhood"
    ROAD = "Road"
    INTERSECTION = "Intersection"
    DELAY = "Delay"
    ACCIDENT = "Accident"
    CONSTRUCTION = "Construction"
    POLICY = "Policy"
    SERVICE = "Service"
    ORGANIZATION = "Organization"
    DATE = "Date"
    SCHEDULE = "Schedule"
    DOCUMENT = "Document"

class RelationType(str, Enum):
    """Relationship types for Neo4j"""
    # Structural
    INCLUDES_STATION = "INCLUDES_STATION"
    STOPS_AT = "STOPS_AT"
    CONNECTS_TO = "CONNECTS_TO"
    INTERCHANGE_WITH = "INTERCHANGE_WITH"
    PART_OF = "PART_OF"
    LOCATED_IN = "LOCATED_IN"

    # Operational
    OPERATES = "OPERATES"
    MANAGES = "MANAGES"
    PROVIDED_BY = "PROVIDED_BY"
    INTERSECTS = "INTERSECTS"
    CONNECTS = "CONNECTS"

    # Incident
    AFFECTS = "AFFECTS"
    OCCURRED_AT = "OCCURRED_AT"
    BLOCKS = "BLOCKS"
    CAUSES = "CAUSES"
    OCCURRED_ON = "OCCURRED_ON"

    # Policy
    IMPLEMENTS = "IMPLEMENTS"
    APPLIES_TO = "APPLIES_TO"
    FUNDED_BY = "FUNDED_BY"

    # Document
    MENTIONS = "MENTIONS"
    ABOUT = "ABOUT"
    DESCRIBES = "DESCRIBES"
    EXTRACTED_FROM = "EXTRACTED_FROM"

class Neo4jEntity(BaseModel):
    """Entity for Neo4j ingestion"""
    id: str = Field(..., description="Unique entity ID")
    label: EntityType = Field(..., description="Neo4j node label")
    properties: Dict[str, Any] = Field(..., description="Node properties")

    def to_cypher_create(self) -> str:
        """Generate Cypher CREATE statement"""
        props_str = ", ".join([f"{k}: ${k}" for k in self.properties.keys()])
        return f"CREATE (e:{self.label.value} {{id: $id, {props_str}}})"

    def to_cypher_merge(self) -> str:
        """Generate Cypher MERGE statement (upsert)"""
        props_str = ", ".join([f"e.{k} = ${k}" for k in self.properties.keys()])
        return f"""
        MERGE (e:{self.label.value} {{id: $id}})
        ON CREATE SET {props_str}
        ON MATCH SET {props_str}
        """

class Neo4jRelation(BaseModel):
    """Relationship for Neo4j ingestion"""
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = Field(default_factory=dict)

    def to_cypher(self) -> str:
        """Generate Cypher relationship creation"""
        props_str = ", ".join([f"{k}: ${k}" for k in self.properties.keys()])
        props_clause = f" {{{props_str}}}" if props_str else ""

        return f"""
        MATCH (source {{id: $source_id}})
        MATCH (target {{id: $target_id}})
        MERGE (source)-[r:{self.relation_type.value}{props_clause}]->(target)
        """

class KnowledgeGraphForNeo4j(BaseModel):
    """Complete KG ready for Neo4j ingestion"""
    document_url: str
    document_title: str
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    entities: List[Neo4jEntity]
    relations: List[Neo4jRelation]
    summary: str
```

---

## 🚀 Implementation Examples

### 1. Entity Extraction with LLM

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import List

class ExtractedEntity(BaseModel):
    """LLM extraction output format"""
    name: str
    entity_type: str  # Will map to EntityType
    description: str
    properties: Dict[str, Any] = {}

class ExtractedRelation(BaseModel):
    """LLM extraction output format"""
    source_entity: str  # Entity name
    target_entity: str  # Entity name
    relation_type: str  # Will map to RelationType
    description: str

class ExtractionResult(BaseModel):
    """Complete LLM extraction result"""
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]
    summary: str

async def extract_knowledge_from_markdown(
    markdown_content: str,
    document_url: str,
    llm_client: OpenAI
) -> ExtractionResult:
    """Extract entities and relations using LLM structured output"""

    prompt = f"""Extract knowledge from this Seoul traffic documentation.

Focus on:
- Transport entities: subway lines, bus routes, stations, terminals
- Location entities: districts, neighborhoods, roads, intersections
- Incident entities: delays, accidents, construction projects
- Policy entities: policies, services, organizations
- Temporal entities: dates, schedules

Extract meaningful relationships between entities.

Document:
{markdown_content}

Return structured JSON with entities and relations."""

    response = llm_client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a transportation domain knowledge extraction expert."},
            {"role": "user", "content": prompt}
        ],
        response_format=ExtractionResult
    )

    return response.choices[0].message.parsed
```

### 2. Neo4j Ingestion

```python
class KnowledgeGraphIngester:
    """Ingest extracted knowledge into Neo4j"""

    def __init__(self, neo4j_kg: Neo4jKnowledgeGraph):
        self.kg = neo4j_kg

    def ingest_extraction(
        self,
        extraction: ExtractionResult,
        document_url: str
    ) -> None:
        """Ingest LLM extraction results into Neo4j"""

        with self.kg.driver.session(database=self.kg.database) as session:
            # 1. Create document node
            session.run("""
                MERGE (d:Document {url: $url})
                SET d.title = $title,
                    d.summary = $summary,
                    d.updated_at = datetime()
            """, url=document_url, title="...", summary=extraction.summary)

            # 2. Create entities
            entity_id_map = {}  # Map entity names to IDs
            for entity in extraction.entities:
                entity_id = self._generate_entity_id(entity.name, entity.entity_type)
                entity_id_map[entity.name] = entity_id

                neo4j_entity = Neo4jEntity(
                    id=entity_id,
                    label=EntityType(entity.entity_type),
                    properties={
                        "name": entity.name,
                        "description": entity.description,
                        **entity.properties,
                        "source_url": document_url,
                        "created_at": datetime.utcnow()
                    }
                )

                session.run(
                    neo4j_entity.to_cypher_merge(),
                    id=neo4j_entity.id,
                    **neo4j_entity.properties
                )

                # Link entity to document
                session.run("""
                    MATCH (e {id: $entity_id})
                    MATCH (d:Document {url: $doc_url})
                    MERGE (e)-[:EXTRACTED_FROM]->(d)
                """, entity_id=entity_id, doc_url=document_url)

            # 3. Create relationships
            for relation in extraction.relations:
                source_id = entity_id_map.get(relation.source_entity)
                target_id = entity_id_map.get(relation.target_entity)

                if source_id and target_id:
                    neo4j_relation = Neo4jRelation(
                        source_id=source_id,
                        target_id=target_id,
                        relation_type=RelationType(relation.relation_type),
                        properties={
                            "description": relation.description,
                            "source_url": document_url
                        }
                    )

                    session.run(
                        neo4j_relation.to_cypher(),
                        source_id=source_id,
                        target_id=target_id,
                        **neo4j_relation.properties
                    )

    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate consistent entity ID"""
        import hashlib
        name_normalized = name.lower().strip()
        hash_input = f"{entity_type}:{name_normalized}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
```

### 3. Cypher Query Examples

```python
class Neo4jQueryInterface:
    """Query interface for knowledge graph"""

    def find_station_connections(self, station_name: str) -> List[Dict]:
        """Find all subway lines and bus routes at a station"""
        query = """
        MATCH (s:Station {name: $station_name})
        OPTIONAL MATCH (s)<-[:INCLUDES_STATION]-(line:SubwayLine)
        OPTIONAL MATCH (s)<-[:STOPS_AT]-(bus:BusRoute)
        RETURN s.name as station,
               collect(DISTINCT line.name) as subway_lines,
               collect(DISTINCT bus.name) as bus_routes
        """
        with self.kg.driver.session() as session:
            result = session.run(query, station_name=station_name)
            return [dict(record) for record in result]

    def find_incident_impact(self, incident_id: str) -> List[Dict]:
        """Find cascading effects of an incident"""
        query = """
        MATCH (i:Delay|Accident {id: $incident_id})-[:AFFECTS]->(affected)
        OPTIONAL MATCH (i)-[:CAUSES]->(caused:Delay)
        RETURN i.description as incident,
               collect(DISTINCT affected.name) as affected_entities,
               collect(DISTINCT caused.description) as cascading_effects
        """
        with self.kg.driver.session() as session:
            result = session.run(query, incident_id=incident_id)
            return [dict(record) for record in result]

    def find_policy_coverage(self, policy_id: str) -> List[Dict]:
        """Find all transport services covered by a policy"""
        query = """
        MATCH (p:Policy {id: $policy_id})-[:APPLIES_TO]->(transport)
        WHERE transport:SubwayLine OR transport:BusRoute
        RETURN p.name as policy,
               collect({
                   type: labels(transport)[0],
                   name: transport.name
               }) as covered_services
        """
        with self.kg.driver.session() as session:
            result = session.run(query, policy_id=policy_id)
            return [dict(record) for record in result]

    def vector_search_documents(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict]:
        """Semantic search using vector embeddings"""
        query = """
        CALL db.index.vector.queryNodes(
            'document_embeddings',
            $top_k,
            $query_embedding
        ) YIELD node, score
        RETURN node.url as url,
               node.title as title,
               node.summary as summary,
               score
        ORDER BY score DESC
        """
        with self.kg.driver.session() as session:
            result = session.run(
                query,
                query_embedding=query_embedding,
                top_k=top_k
            )
            return [dict(record) for record in result]
```

---

## 📈 Updated Implementation Roadmap

### Phase 1: Setup & MVP (Week 1)

**Day 1-2: Neo4j Setup**
- [ ] Install Neo4j Desktop or Docker container
- [ ] Add dependencies: `uv add neo4j neo4j-graphrag-python`
- [ ] Create connection utilities
- [ ] Define schema and create indexes
- [ ] Test basic CRUD operations

**Day 3-5: Entity Extraction**
- [ ] Implement Pydantic models for entities/relations
- [ ] Create LLM extraction with OpenAI structured output
- [ ] Test extraction on 5 sample documents
- [ ] Validate extraction quality

**Day 6-7: Graph Ingestion**
- [ ] Implement entity resolution (deduplication)
- [ ] Create Cypher ingestion pipeline
- [ ] Ingest 10 sample documents
- [ ] Verify graph structure in Neo4j Browser

**Deliverables:**
- `src/knowledge/neo4j_connector.py`
- `src/knowledge/schemas.py`
- `src/knowledge/extractor.py`
- `src/knowledge/ingester.py`
- 10 documents in Neo4j graph

### Phase 2: GraphRAG Integration (Week 2)

**Day 1-3: Vector Embeddings**
- [ ] Generate embeddings for documents and chunks
- [ ] Create vector index in Neo4j
- [ ] Implement hybrid search (vector + graph)

**Day 4-5: Query Interface**
- [ ] Implement Cypher query templates
- [ ] Create natural language to Cypher converter
- [ ] Build GraphRAG retrieval pipeline

**Day 6-7: Visualization**
- [ ] Set up Neo4j Bloom or custom viz
- [ ] Create dashboard for graph exploration
- [ ] Generate sample query examples

**Deliverables:**
- `src/knowledge/embeddings.py`
- `src/knowledge/query_interface.py`
- `src/knowledge/graphrag_retriever.py`
- Interactive graph explorer

### Phase 3: Production Scale (Week 3-4)

**Day 1-5: Batch Processing**
- [ ] Process all 59+ documents
- [ ] Implement incremental updates
- [ ] Add entity resolution across documents
- [ ] Quality validation and metrics

**Day 6-10: Optimization**
- [ ] Query performance tuning
- [ ] Index optimization
- [ ] Batch ingestion optimization
- [ ] Cost analysis (LLM API calls)

**Day 11-14: Application Layer**
- [ ] Build FAQ generation from graph
- [ ] Implement semantic search API
- [ ] Create timeline view for policies
- [ ] Impact analysis tool

**Deliverables:**
- `src/scripts/batch_knowledge_extraction.py`
- Complete knowledge graph (all documents)
- Performance metrics report
- Application API endpoints

---

## 🎯 Success Metrics

### Graph Metrics
- **Node Count:** 500-1,000 entities
- **Relationship Count:** 2,000-5,000 relations
- **Average Connectivity:** 3-5 relationships per entity
- **Extraction Accuracy:** 85-90% (with confidence > 0.7)

### Query Performance
- **Simple Queries:** < 50ms
- **Graph Traversal:** < 200ms
- **Vector Search:** < 100ms
- **Complex Analytics:** < 1s

### Cost Metrics
- **LLM Cost per Document:** $0.01-0.05
- **Total Extraction Cost:** $3-15 (59 docs * $0.05)
- **Neo4j Storage:** < 1GB for full graph

---

## 📚 Resources

**Neo4j Official:**
- Neo4j GraphRAG Python: https://neo4j.com/docs/neo4j-graphrag-python/
- LLM Graph Builder: https://neo4j.com/labs/genai-ecosystem/llm-graph-builder/
- Cypher Manual: https://neo4j.com/docs/cypher-manual/

**Best Practices:**
- GraphRAG with Neo4j: https://www.analyticsvidhya.com/blog/2024/11/graphrag-with-neo4j/
- Knowledge Graph RAG: https://neo4j.com/blog/developer/knowledge-graph-rag-application/

**LangChain Integration:**
- Neo4j LangChain: https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/

---

**Next Step:** Set up Neo4j and begin Phase 1 implementation!
