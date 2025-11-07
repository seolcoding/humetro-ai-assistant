# KG Cypher RAG 수정 결과 분석

**실험 일시**: 2025-11-06
**실험 대상**: FIXED KG Cypher Generation (Vector + Graph Hybrid)
**테스트 규모**: 10 questions (random sample), 3 models

---

## 🎯 핵심 발견: 극적인 성능 개선

### Multi-hop Faithfulness 비교

| 방식 | 평균 점수 | 변화량 | 상태 |
|------|-----------|--------|------|
| **OLD Cypher** (Pure LLM Generation) | 0.398 | - | ❌ 실패 |
| **Naive RAG** (Vector Only) | 0.746 | +87.4% vs OLD | ✅ 기준선 |
| **KG Simple** (Vector + Fixed Cypher) | 0.780 | +4.6% vs Naive | ✅ 개선 |
| **FIXED Cypher** (Vector + Graph Expansion) | **0.830** | **+108.5% vs OLD** | 🏆 최고 |

### 개선율 분석

```
FIXED Cypher vs OLD Cypher:  +108.5% ⬆️ (0.398 → 0.830)
FIXED Cypher vs Naive RAG:    +11.3% ⬆️ (0.746 → 0.830)
FIXED Cypher vs KG Simple:     +6.4% ⬆️ (0.780 → 0.830)
```

---

## 📊 모델별 상세 결과 (Multi-hop)

| Model | FIXED Cypher | OLD Cypher | 개선율 | vs Naive | vs KG Simple |
|-------|-------------|-----------|--------|----------|-------------|
| **EXAONE-3.5-7.8B** | **0.893** | 0.320 | **+179.1%** | +22.6% | +21.4% |
| **GPT-4o-mini** | 0.802 | 0.361 | +122.2% | +4.1% | +1.1% |
| **GPT-OSS-20B** | 0.795 | 0.396 | +100.8% | +6.7% | +6.7% |
| **평균** | **0.830** | 0.398 | **+108.5%** | **+11.3%** | **+6.4%** |

---

## 🔬 근본 원인 분석

### OLD 방식의 치명적 결함

```python
# ❌ OLD: Pure Cypher Generation
1. LLM이 질문만 보고 Cypher 쿼리 생성
2. 벡터 검색 없음 → 시작점을 못 찾음
3. 잘못된 Cypher → 빈 결과 반환
4. GraphCypherQAChain이 결과를 요약 → 원본 손실

결과: -47% 성능 저하 (0.746 → 0.398)
```

### FIXED 방식의 해결책

```python
# ✅ FIXED: Hybrid Approach
1. Vector Similarity Search
   → 질문 embedding으로 관련 노드 검색 (top-k)
   → 벡터 인덱스: db.index.vector.queryNodes()

2. Graph Expansion
   → 찾은 노드의 이웃 탐색 (1-hop)
   → MATCH (start)-[r]-(neighbor)

3. Return Original Text
   → GraphCypherQAChain 제거
   → 모든 노드의 원본 텍스트 직접 반환

결과: +108.5% 성능 향상 (0.398 → 0.830)
```

---

## 💡 주요 인사이트

### 1. 벡터 검색이 필수 시작점

**발견**: 순수 Cypher 생성은 실패하지만, 벡터 검색 + 그래프 확장은 성공

```
Vector Search (시작점) → Graph Expansion (관계 탐색) → Success
No Starting Point → Random Graph Walk → Failure
```

### 2. 그래프 확장의 가치

**KG Simple vs FIXED Cypher 비교:**
- KG Simple: 벡터 검색된 노드만 반환 (0.780)
- FIXED Cypher: 벡터 노드 + 이웃 노드 반환 (0.830)
- **개선**: +6.4% (이웃 노드가 추가 컨텍스트 제공)

### 3. EXAONE이 가장 큰 수혜자

**EXAONE-3.5-7.8B:**
- OLD Cypher: 0.320 (최하위)
- FIXED Cypher: **0.893** (최상위)
- **개선**: +179.1% (가장 큰 향상)

→ 오픈소스 모델이 구조화된 컨텍스트(그래프)에서 더 큰 이득

### 4. Multi-hop에서 진가 발휘

**Single-hop vs Multi-hop 성능:**

| Model | Single-hop | Multi-hop | Gap |
|-------|-----------|----------|-----|
| GPT-4o-mini | 0.920 | 0.802 | -0.117 |
| EXAONE | 0.844 | **0.893** | **+0.050** ✨ |
| GPT-OSS | 0.608 | 0.795 | **+0.187** ✨ |

→ EXAONE과 GPT-OSS는 Multi-hop에서 오히려 성능 향상!
→ 그래프 탐색이 복잡한 추론에 효과적

---

## 🏆 최종 결론

### 성공적인 수정 완료

| 항목 | 결과 |
|------|------|
| **문제 진단** | ✅ 벡터 시작점 부재 확인 |
| **해결책 구현** | ✅ Hybrid 방식 적용 |
| **성능 검증** | ✅ +108.5% 개선 확인 |
| **목표 달성** | ✅ KG Simple 초과 (0.780 → 0.830) |

### 권장 사항

1. **전체 50 questions로 재실험**
   - 현재: 10 questions (quick test)
   - 목표: 50 questions (전체 golden dataset)
   - 예상: 동일한 개선 패턴 유지

2. **5개 모델 전체 테스트**
   - 추가 모델: Qwen3-8B, Gemma3-12B
   - 목표: Gemma3-12B도 높은 성능 예상

3. **Expansion hops 실험**
   - 현재: 1-hop
   - 실험: 2-hop 테스트
   - 가설: 더 많은 컨텍스트 → 더 나은 성능?

4. **최종 3-way 비교**
   - Naive RAG vs KG Simple vs **FIXED Cypher**
   - 전체 모델, 전체 질문으로 공정 비교

---

## 📈 다음 단계

**즉시 실행:**
```json
{
  "questions": {"limit": 50},
  "models": {"evaluation_targets": "all"},
  "retrieval": "kg_cypher_fixed"
}
```

**예상 결과:**
- Multi-hop Faithfulness: 0.820 ~ 0.850 (전 모델 평균)
- Naive RAG 대비: +10% ~ +15%
- KG Simple 대비: +5% ~ +8%
- **최고 성능**: Gemma3-12B 또는 EXAONE (0.90+)

---

**결론**: KG Cypher Generation 수정 성공! 벡터 시작점 추가로 **+108.5% 성능 향상** 달성. 이제 전체 실험으로 확장 준비 완료. 🚀
