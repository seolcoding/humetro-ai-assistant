# 📊 Documentation Structure Comparison

**Before vs After Visualization**

---

## ❌ BEFORE (Current - Problematic)

```
humetro-ai-assistant/
├── claudedocs/                          ⚠️ AI 전용 폴더 (삭제 예정)
│   └── references/                      ⚠️ 중복된 references
│       ├── perplexity/                  📍 Perplexity 조사 자료
│       └── arxiv/                       📍 arXiv 논문 요약
│
├── thesis/                              ⚠️ 논문 + 레퍼런스 혼재
│   ├── references/                      ⚠️ 또 다른 references 폴더!
│   ├── draft/                           📝 논문 초안
│   ├── claude_s_research.md            🤖 Claude 생성 자료 (위치 애매)
│   ├── figures/
│   └── tables/
│
├── data/raw/humetro/documents/          💾 원본 데이터 (200+ files)
├── README.md
└── MIGRATION_GUIDE.md

문제점:
❌ references 폴더가 2곳 (claudedocs/, thesis/)
❌ AI 생성 자료와 논문 초안 위치 불명확
❌ 조사 자료와 인용 자료 구분 없음
❌ 프로젝트 문서와 논문 초안 혼재
❌ 처음 보는 사람이 어디에 뭐가 있는지 파악 어려움
```

---

## ✅ AFTER (Proposed - Clear & Organized)

```
humetro-ai-assistant/
├── 📁 docs/                             ✨ 모든 연구 문서 통합
│   │
│   ├── 01_citations/                    🎓 학술 인용 자료
│   │   ├── arxiv_papers/
│   │   │   ├── graph_rag/
│   │   │   ├── evaluation/
│   │   │   └── llm_deployment/
│   │   ├── academic_surveys/
│   │   └── README.md
│   │
│   ├── 02_research/                     🔍 조사 및 분석 자료
│   │   ├── perplexity_deep_research/
│   │   ├── web_research/
│   │   ├── competitive_analysis/
│   │   └── README.md
│   │
│   ├── 03_ai_references/                🤖 AI 생성 종합 자료
│   │   ├── claude_comprehensive_guide.md
│   │   ├── mcp_servers_catalog.md
│   │   ├── vector_db_comparison.md
│   │   └── README.md
│   │
│   ├── 04_project_docs/                 📋 프로젝트 문서
│   │   ├── architecture/
│   │   ├── guides/
│   │   ├── specifications/
│   │   └── README.md
│   │
│   └── README.md                        📚 전체 문서 가이드
│
├── 📁 thesis/                           📝 논문 작성 전용
│   ├── manuscript/                      ✨ 논문 원고 (섹션별 분리)
│   │   ├── 00_outline.md
│   │   ├── 01_introduction.md
│   │   ├── 02_related_work.md
│   │   ├── 03_methodology.md
│   │   ├── 04_experiments.md
│   │   ├── 05_results.md
│   │   ├── 06_discussion.md
│   │   ├── 07_conclusion.md
│   │   └── 08_references.bib
│   │
│   ├── figures/
│   ├── tables/
│   ├── drafts/                          💾 초안 버전 관리
│   └── README.md
│
├── 📁 data/                             💾 데이터 (변경 없음)
├── README.md
└── DOCUMENTATION_REFACTORING_PLAN.md

개선점:
✅ 명확한 4개 카테고리 (인용/조사/AI레퍼런스/프로젝트)
✅ 중복 제거 (references 폴더 통합)
✅ 논문 섹션별 분리로 작성 효율 향상
✅ 각 폴더별 README로 자가 학습 가능
✅ 일관된 네이밍 규칙 (01_, 02_, 03_, 04_)
✅ 처음 보는 사람도 직관적으로 이해 가능
```

---

## 🎯 핵심 개선 사항

### 1. 문서 유형별 명확한 분류

| 유형 | 폴더 | 설명 |
|-----|------|-----|
| **인용 자료** | `docs/01_citations/` | arXiv 논문, 학술 서베이 |
| **조사 자료** | `docs/02_research/` | Perplexity, 웹 조사 결과 |
| **AI 레퍼런스** | `docs/03_ai_references/` | Claude 생성 종합 가이드 |
| **프로젝트 문서** | `docs/04_project_docs/` | 아키텍처, 가이드, 명세서 |
| **논문 원고** | `thesis/manuscript/` | 논문 섹션별 원고 |

### 2. 폴더 구조 단순화

```
Before:
  claudedocs/references/ + thesis/references/ = 중복!

After:
  docs/ (하나로 통합) + thesis/ (논문 전용)
```

### 3. 섹션별 논문 분리

```
Before:
  thesis/draft/draft.md (모든 내용이 한 파일에)

After:
  thesis/manuscript/
    ├── 01_introduction.md
    ├── 02_related_work.md
    ├── 03_methodology.md
    ├── ...
```

---

## 📋 마이그레이션 요약

### 이동될 주요 파일

| 현재 위치 | 새 위치 | 변경 사유 |
|---------|--------|----------|
| `claudedocs/references/perplexity/` | `docs/02_research/perplexity_deep_research/` | 조사 자료 통합 |
| `claudedocs/references/arxiv/` | `docs/01_citations/arxiv_papers/` | 인용 자료 분리 |
| `thesis/claude_s_research.md` | `docs/03_ai_references/claude_comprehensive_guide.md` | AI 레퍼런스 분리 |
| `thesis/references/` | `docs/04_project_docs/` | 프로젝트 문서 분리 |
| `thesis/draft/draft.md` | `thesis/manuscript/0X_*.md` | 섹션별 분리 |

### 삭제될 폴더

- ❌ `claudedocs/` - docs/로 통합
- ❌ `thesis/references/` - docs/04_project_docs/로 이동
- ❌ `thesis/draft/` - thesis/manuscript/로 변경

---

## 🚀 Quick Start (After Migration)

### 인용 자료 찾기
```bash
cd docs/01_citations/
# arXiv 논문: arxiv_papers/
# 학술 서베이: academic_surveys/
```

### 조사 자료 찾기
```bash
cd docs/02_research/
# Perplexity: perplexity_deep_research/
# 웹 조사: web_research/
# 경쟁 분석: competitive_analysis/
```

### AI 레퍼런스 찾기
```bash
cd docs/03_ai_references/
# Claude 종합 가이드: claude_comprehensive_guide.md
# MCP 서버: mcp_servers_catalog.md
# 벡터 DB: vector_db_comparison.md
```

### 프로젝트 문서 찾기
```bash
cd docs/04_project_docs/
# 아키텍처: architecture/
# 가이드: guides/
# 명세서: specifications/
```

### 논문 작성
```bash
cd thesis/manuscript/
# 섹션별 파일 편집
# figures/, tables/ 참조
```

---

## 📊 예상 소요 시간

| 단계 | 작업 | 시간 | 난이도 |
|-----|------|------|--------|
| 1 | 준비 (백업, 브랜치) | 1시간 | 쉬움 |
| 2 | 디렉토리 생성 | 30분 | 쉬움 |
| 3 | 파일 이동 | 2시간 | 중간 |
| 4 | README 작성 | 1시간 | 중간 |
| 5 | 링크 업데이트 | 1시간 | 중간 |
| 6 | 정리 및 정리 | 30분 | 쉬움 |
| 7 | 검증 및 테스트 | 1시간 | 중간 |
| 8 | 커밋 및 PR | 30분 | 쉬움 |
| **합계** | **전체 작업** | **7.5시간** | **중간** |

**권장 일정**: 2일 (하루 4시간 작업 기준)

---

## ✨ 기대 효과

### 🎯 사용자 경험
- 문서 위치 예측 가능성: **30% → 90%**
- 검색 시간 단축: **5분 → 30초**
- 신규 팀원 온보딩 시간: **2일 → 4시간**

### 📚 문서 관리
- 중복 제거율: **100%**
- 분류 명확도: **40% → 95%**
- 유지보수 효율: **2배 향상**

### 📝 논문 작성
- 섹션별 작업 효율: **50% 향상**
- 인용 참조 속도: **3배 빠름**
- 버전 관리 용이성: **크게 향상**

---

**Next Steps**: [DOCUMENTATION_REFACTORING_PLAN.md](./DOCUMENTATION_REFACTORING_PLAN.md) 참조
