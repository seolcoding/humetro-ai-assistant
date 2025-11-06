# Knowledge Graph Data Quality Limitations

## Overview
This document describes known data quality issues in the Seoul Traffic Knowledge Graph that affect RAG retrieval performance.

## Critical Limitations

### 1. Date Information Incomplete on Connected Entities

**Issue**: Traffic Incident nodes with route connections rarely have dates populated.

**Impact**:
- Date-specific queries (e.g., "2024-12-14에 버스 우회") return empty results
- LLM generates valid Cypher but graph data lacks the expected attribute

**Data Pattern**:
```cypher
# What DOESN'T work:
MATCH (incident:`Traffic Incident`)-[:OCCURS_ON]->(route:Route)
WHERE incident.date = "2024-12-14"  // Most incidents with routes have date=None

# What DOES work:
MATCH (incident:`Traffic Incident`)-[:OCCURS_ON]->(route:Route)
WHERE route.name IS NOT NULL  // No date filter
```

**Evidence**:
```
Query: MATCH (incident:`Traffic Incident`)-[:OCCURS_ON]->(route:Route)
       WHERE incident.date IS NOT NULL AND route.name IS NOT NULL
Result: 0 rows found

Query: MATCH (incident:`Traffic Incident`)-[:OCCURS_ON]->(route:Route)
       WHERE route.name IS NOT NULL
Result: 10+ rows found (but all have date=None)
```

### 2. Node Labels Contain Spaces

**Issue**: Entity labels include spaces (e.g., `Bus Service`, `Traffic Incident`)

**Impact**:
- LLMs often generate incorrect Cypher without backtick notation
- Requires explicit schema documentation in prompts

**Solution**:
```cypher
# CORRECT:
MATCH (service:`Bus Service`)-[:SERVES]->(route:Route)

# INCORRECT (fails):
MATCH (service:BusService)-[:SERVES]->(route:Route)
```

### 3. Many Nodes Lack Primary Names

**Issue**: Key entities (Route, Traffic Incident) often have empty name/description fields

**Impact**:
- Retrieval returns "N/A" for entity names
- Context quality reduced for LLM answer generation

**Example**:
```
Traffic Incident nodes: 778 total
- With names: <5%
- With descriptions: ~30%
- Empty (no name/description): ~65%
```

### 4. Relationship Coverage Gaps

**Issue**: Expected entity connections are incomplete

**Examples**:
- Stations "영등포역", "여의도" exist but have no `CONNECTS` relationships to routes
- Many Traffic Incidents have dates BUT no route connections
- Many Traffic Incidents have route connections BUT no dates

**Impact**: Complex multi-hop queries often return empty results

## Working Query Patterns

### ✅ Works Well:
1. **Service → Route queries**:
```cypher
MATCH (service:`Bus Service`)-[:SERVES]->(route:Route)
WHERE service.name IS NOT NULL
RETURN service.name, route.name
```

2. **Incident → Route → Service (no date filter)**:
```cypher
MATCH (incident:`Traffic Incident`)-[:OCCURS_ON]->(route:Route)
MATCH (service:`Bus Service`)-[:SERVES]->(route)
WHERE route.name IS NOT NULL
RETURN incident.description, route.name, service.name
```

3. **Station → Line queries**:
```cypher
MATCH (station:Station)-[:IS_PART_OF]->(line:Line)
WHERE station.name IS NOT NULL
RETURN station.name, line.name
```

### ❌ Often Returns Empty:
1. Date-filtered incident queries
2. Station connection queries (CONNECTS relationships)
3. Queries requiring both dates AND routes
4. Multi-hop queries with >3 relationships

## Recommendations

### For Cypher Generation Prompts:
1. ✅ Always use backticks for labels with spaces
2. ✅ Include `IS NOT NULL` checks for name fields
3. ✅ Provide working examples without dates
4. ✅ Document data limitations explicitly
5. ❌ Avoid examples with date filters on Traffic Incidents

### For KG Build Improvements:
1. **Priority 1**: Populate dates on incidents with route connections
2. **Priority 2**: Extract entity names more consistently
3. **Priority 3**: Build CONNECTS relationships for stations
4. **Priority 4**: Enrich route descriptions from source documents

### For RAG Evaluation:
1. Use questions that match working patterns
2. Don't expect date-specific answers for incidents
3. Focus on service-route-incident relationships (without dates)
4. Acknowledge empty results may be data issues, not retrieval failures

## Statistics

### Complete Paths Found:
- `Traffic Incident → Route ← Bus Service`: 5 complete paths (0 with dates)
- `Station → Line`: 671 stations (good coverage)
- `Bus Service → Route`: 10+ services with routes

### Empty Result Patterns:
- Incidents with dates AND routes: 0
- Stations with CONNECTS relationships: 0 for tested stations
- Named routes on dated incidents: 0

## Conclusion

The KG has valuable structure but incomplete data population. RAG retrieval works best for:
- General service/route information
- Incident patterns (without specific dates)
- Transportation policies and reports

Date-specific queries and multi-hop station routing currently have limited utility due to data gaps.

---
**Last Updated**: 2025-11-06
**KG Build Date**: 2025-11-05
**Total Nodes**: 6,544
**Total Relationships**: 9,554
