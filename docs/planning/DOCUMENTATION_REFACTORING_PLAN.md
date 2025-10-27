# 📚 Documentation Refactoring Plan

**Created**: 2025-10-27
**Purpose**: 리파지토리 문서 체계 재정리 - 인용/조사/AI 레퍼런스/논문 본문 분리
**Status**: Planning Phase

---

## 🎯 목표 (Goals)

1. **명확한 문서 분류**: 용도별로 문서를 명확하게 구분
2. **직관적인 구조**: 처음 보는 사람도 쉽게 이해할 수 있는 계층 구조
3. **중복 제거**: 유사한 내용의 문서 통합 및 정리
4. **접근성 향상**: 각 문서의 목적과 위치를 쉽게 파악

---

## 📊 현재 상태 분석 (Current State Analysis)

### 기존 문서 구조 문제점

```
현재 구조의 문제:
├── claudedocs/references/          # AI가 생성한 조사 자료
│   ├── perplexity/                 # Perplexity 검색 결과
│   └── arxiv/                      # arXiv 논문 요약
│
├── thesis/                         # 논문 관련 자료
│   ├── references/                 # 또 다른 레퍼런스 폴더 (중복!)
│   ├── draft/                      # 논문 초안
│   ├── claude_s_research.md        # Claude 조사 자료 (위치 애매)
│   └── figures/, tables/           # 논문 자산
│
└── data/raw/humetro/documents/     # 원본 데이터 (문서와 혼재)

❌ 문제점:
1. references 폴더가 2개 (claudedocs/, thesis/)
2. AI 생성 자료와 논문 초안이 혼재
3. 원본 데이터와 연구 문서 경계 불명확
4. 문서 유형별 구분이 없음
```

### 문서 유형별 분류

| 유형 | 설명 | 현재 위치 | 파일 수 |
|-----|------|---------|--------|
| **인용 자료** | 학술 논문, arXiv 원본 | claudedocs/references/arxiv/ | ~10개 |
| **조사 자료** | Perplexity, 웹 검색 결과 | claudedocs/references/perplexity/ | ~4개 |
| **AI 레퍼런스** | Claude 생성 종합 자료 | thesis/claude_s_research.md | 1개 |
| **논문 초안** | 작성 중인 논문 본문 | thesis/draft/ | ~5개 |
| **프로젝트 문서** | 프로젝트 설명, 가이드 | thesis/references/ | ~6개 |
| **원본 데이터** | 다산콜센터, Humetro 문서 | data/raw/ | 200+ |

---

## 🎨 새로운 문서 구조 (Proposed Structure)

### 핵심 원칙

1. **Top-level 단순화**: 최상위는 3개 카테고리만 (docs/, thesis/, data/)
2. **명확한 네이밍**: 폴더명으로 내용 유추 가능
3. **논리적 계층**: 용도 → 출처 → 주제 순서
4. **중복 제거**: 단일 진실 공급원 (Single Source of Truth)

### 제안된 구조

```
humetro-ai-assistant/
│
├── 📁 docs/                                    # 📚 모든 연구 문서 (통합)
│   │
│   ├── 📂 01_citations/                        # 🎓 학술 인용 자료
│   │   ├── arxiv_papers/                      # arXiv 원본 논문 요약
│   │   │   ├── graph_rag/                     # Graph RAG 관련
│   │   │   ├── evaluation/                    # 평가 메트릭 관련
│   │   │   └── llm_deployment/                # LLM 배포 관련
│   │   │
│   │   ├── academic_surveys/                  # 서베이 논문 정리
│   │   │   ├── rag_survey_2024.md
│   │   │   ├── knowledge_graph_survey.md
│   │   │   └── llm_evaluation_survey.md
│   │   │
│   │   └── README.md                          # 인용 자료 가이드
│   │
│   ├── 📂 02_research/                         # 🔍 조사 및 분석 자료
│   │   ├── perplexity_deep_research/          # Perplexity 심층 조사
│   │   │   ├── 01_graph_rag_fundamentals.md
│   │   │   ├── 02_ragas_evaluation.md
│   │   │   ├── 03_opensource_llm_comparison.md
│   │   │   └── 04_cost_analysis.md
│   │   │
│   │   ├── web_research/                      # 웹 검색 기반 조사
│   │   │   ├── neo4j_integration.md
│   │   │   ├── korean_nlp_resources.md
│   │   │   └── mcp_servers_comparison.md
│   │   │
│   │   ├── competitive_analysis/              # 경쟁 기술 분석
│   │   │   ├── api_vs_onpremise.md
│   │   │   └── rag_frameworks_comparison.md
│   │   │
│   │   └── README.md                          # 조사 자료 가이드
│   │
│   ├── 📂 03_ai_references/                    # 🤖 AI 생성 종합 자료
│   │   ├── claude_comprehensive_guide.md      # Claude 종합 가이드 (기존 claude_s_research.md)
│   │   ├── mcp_servers_catalog.md             # MCP 서버 카탈로그
│   │   ├── vector_db_comparison.md            # 벡터 DB 비교
│   │   ├── implementation_patterns.md         # 구현 패턴 모음
│   │   └── README.md                          # AI 레퍼런스 가이드
│   │
│   ├── 📂 04_project_docs/                     # 📋 프로젝트 문서
│   │   ├── architecture/                      # 아키텍처 문서
│   │   │   ├── system_design.md
│   │   │   ├── data_pipeline.md
│   │   │   └── evaluation_framework.md
│   │   │
│   │   ├── guides/                            # 가이드 문서
│   │   │   ├── setup_guide.md
│   │   │   ├── development_workflow.md
│   │   │   └── deployment_guide.md
│   │   │
│   │   ├── specifications/                    # 상세 명세서
│   │   │   ├── rag_pipeline_spec.md
│   │   │   ├── api_spec.md
│   │   │   └── data_format_spec.md
│   │   │
│   │   └── README.md                          # 프로젝트 문서 가이드
│   │
│   └── README.md                              # docs 폴더 메인 가이드
│
├── 📁 thesis/                                  # 📝 논문 작성 영역
│   │
│   ├── 📂 manuscript/                          # 📄 논문 원고
│   │   ├── 00_outline.md                      # 논문 개요
│   │   ├── 01_introduction.md                 # 서론
│   │   ├── 02_related_work.md                 # 관련 연구
│   │   ├── 03_methodology.md                  # 연구 방법
│   │   ├── 04_experiments.md                  # 실험 및 평가
│   │   ├── 05_results.md                      # 결과 및 분석
│   │   ├── 06_discussion.md                   # 논의
│   │   ├── 07_conclusion.md                   # 결론
│   │   └── 08_references.bib                  # 참고문헌 (BibTeX)
│   │
│   ├── 📂 figures/                             # 그림 파일
│   │   ├── architecture_diagram.png
│   │   ├── performance_comparison.png
│   │   └── cost_analysis_chart.png
│   │
│   ├── 📂 tables/                              # 표 데이터
│   │   ├── model_comparison.csv
│   │   ├── evaluation_results.csv
│   │   └── cost_breakdown.csv
│   │
│   ├── 📂 drafts/                              # 초안 및 버전 관리
│   │   ├── v1_initial_draft.md
│   │   ├── v2_advisor_review.md
│   │   └── final_submission/
│   │
│   └── README.md                              # 논문 작성 가이드
│
├── 📁 data/                                    # 💾 데이터 자산 (변경 없음)
│   ├── raw/                                   # 원본 데이터
│   ├── processed/                             # 전처리 데이터
│   └── knowledge_graphs/                      # 지식 그래프
│
├── 📁 src/                                     # 💻 소스 코드 (변경 없음)
├── 📁 experiments/                             # 🧪 실험 관리 (변경 없음)
├── 📁 results/                                 # 📊 실험 결과 (변경 없음)
├── 📁 scripts/                                 # 🔧 자동화 스크립트 (변경 없음)
│
├── README.md                                  # 프로젝트 메인 README
└── DOCUMENTATION_REFACTORING_PLAN.md          # 이 문서

```

---

## 🔄 마이그레이션 매핑 (Migration Mapping)

### Phase 1: 인용 자료 (Citations)

| 현재 경로 | 새 경로 | 작업 |
|---------|--------|-----|
| `claudedocs/references/arxiv/graph_rag_papers.md` | `docs/01_citations/arxiv_papers/graph_rag/comprehensive_survey.md` | 이동 + 리네이밍 |
| *(신규)* | `docs/01_citations/academic_surveys/rag_survey_2024.md` | 새로 생성 |
| *(신규)* | `docs/01_citations/README.md` | 인용 가이드 생성 |

### Phase 2: 조사 자료 (Research)

| 현재 경로 | 새 경로 | 작업 |
|---------|--------|-----|
| `claudedocs/references/perplexity/01_graph_rag_research.md` | `docs/02_research/perplexity_deep_research/01_graph_rag_fundamentals.md` | 이동 + 리네이밍 |
| `claudedocs/references/perplexity/02_ragas_evaluation_metrics.md` | `docs/02_research/perplexity_deep_research/02_ragas_evaluation.md` | 이동 + 리네이밍 |
| `claudedocs/references/perplexity/03_opensource_llm_comparison.md` | `docs/02_research/perplexity_deep_research/03_opensource_llm_comparison.md` | 이동 |
| `claudedocs/references/perplexity/04_onpremise_llm_deployment_cost.md` | `docs/02_research/perplexity_deep_research/04_cost_analysis.md` | 이동 + 리네이밍 |
| `claudedocs/references/README.md` | `docs/02_research/README.md` | 이동 + 업데이트 |

### Phase 3: AI 레퍼런스 (AI References)

| 현재 경로 | 새 경로 | 작업 |
|---------|--------|-----|
| `thesis/claude_s_research.md` | `docs/03_ai_references/claude_comprehensive_guide.md` | 이동 + 리네이밍 |
| *(MCP 서버 정보 추출)* | `docs/03_ai_references/mcp_servers_catalog.md` | 분리 생성 |
| *(벡터 DB 정보 추출)* | `docs/03_ai_references/vector_db_comparison.md` | 분리 생성 |
| *(신규)* | `docs/03_ai_references/README.md` | 가이드 생성 |

### Phase 4: 프로젝트 문서 (Project Docs)

| 현재 경로 | 새 경로 | 작업 |
|---------|--------|-----|
| `thesis/references/PROJECT_DOCUMENTATION.md` | `docs/04_project_docs/architecture/system_design.md` | 이동 + 리네이밍 |
| `thesis/references/workflow_summarization.md` | `docs/04_project_docs/guides/development_workflow.md` | 이동 + 리네이밍 |
| `thesis/references/humetro_rag_blog_post.md` | `docs/04_project_docs/guides/rag_implementation.md` | 이동 + 리네이밍 |
| `MIGRATION_GUIDE.md` | `docs/04_project_docs/guides/migration_guide.md` | 이동 |
| *(신규)* | `docs/04_project_docs/README.md` | 가이드 생성 |

### Phase 5: 논문 원고 (Thesis Manuscript)

| 현재 경로 | 새 경로 | 작업 |
|---------|--------|-----|
| `thesis/draft/draft.md` | `thesis/manuscript/00_outline.md` | 이동 + 섹션 분리 |
| *(draft.md에서 추출)* | `thesis/manuscript/01_introduction.md` | 분리 생성 |
| *(draft.md에서 추출)* | `thesis/manuscript/02_related_work.md` | 분리 생성 |
| `thesis/references/intro.md` | `thesis/manuscript/drafts/v1_introduction.md` | 이동 (초안 보관) |
| `thesis/figures/` | `thesis/figures/` | 그대로 유지 |
| `thesis/tables/` | `thesis/tables/` | 그대로 유지 |

---

## 📋 실행 계획 (Implementation Plan)

### Step 1: 준비 단계 (Preparation)

**기간**: 1일
**목표**: 백업 및 환경 준비

```bash
# 1. 현재 상태 백업
git checkout -b backup-before-doc-refactoring
git add .
git commit -m "docs: backup before documentation refactoring"

# 2. 작업 브랜치 생성
git checkout -b refactor/documentation-structure

# 3. 현재 문서 현황 분석
find . -name "*.md" ! -path "*/.venv/*" ! -path "*/.git/*" > doc_inventory.txt
```

**체크리스트**:

- [ ] Git 백업 브랜치 생성
- [ ] 작업 브랜치 생성
- [ ] 문서 인벤토리 파일 생성
- [ ] 이슈 트래커에 작업 등록

---

### Step 2: 디렉토리 구조 생성 (Directory Creation)

**기간**: 0.5일
**목표**: 새로운 폴더 구조 생성

```bash
# docs/ 폴더 구조 생성
mkdir -p docs/{01_citations,02_research,03_ai_references,04_project_docs}
mkdir -p docs/01_citations/{arxiv_papers,academic_surveys}
mkdir -p docs/01_citations/arxiv_papers/{graph_rag,evaluation,llm_deployment}
mkdir -p docs/02_research/{perplexity_deep_research,web_research,competitive_analysis}
mkdir -p docs/03_ai_references
mkdir -p docs/04_project_docs/{architecture,guides,specifications}

# thesis/ 폴더 구조 정리
mkdir -p thesis/{manuscript,drafts}
```

**체크리스트**:

- [ ] docs/01_citations/ 생성
- [ ] docs/02_research/ 생성
- [ ] docs/03_ai_references/ 생성
- [ ] docs/04_project_docs/ 생성
- [ ] thesis/manuscript/ 생성
- [ ] 각 폴더의 README.md 생성

---

### Step 3: 문서 이동 및 리네이밍 (Migration)

**기간**: 1일
**목표**: 기존 문서를 새 구조로 이동

**우선순위 1 - 조사 자료**:

```bash
# Perplexity 조사 자료 이동
mv claudedocs/references/perplexity/01_graph_rag_research.md \
   docs/02_research/perplexity_deep_research/01_graph_rag_fundamentals.md

mv claudedocs/references/perplexity/02_ragas_evaluation_metrics.md \
   docs/02_research/perplexity_deep_research/02_ragas_evaluation.md

# ... (나머지 파일들)
```

**우선순위 2 - AI 레퍼런스**:

```bash
# Claude 종합 가이드 이동
mv thesis/claude_s_research.md \
   docs/03_ai_references/claude_comprehensive_guide.md
```

**우선순위 3 - 프로젝트 문서**:

```bash
# 프로젝트 문서 이동
mv thesis/references/PROJECT_DOCUMENTATION.md \
   docs/04_project_docs/architecture/system_design.md
```

**체크리스트**:

- [ ] Perplexity 자료 이동 (4개)
- [ ] arXiv 자료 이동 (1개)
- [ ] Claude 레퍼런스 이동 (1개)
- [ ] 프로젝트 문서 이동 (6개)
- [ ] 논문 초안 재구성

---

### Step 4: README 파일 생성 (Documentation)

**기간**: 0.5일
**목표**: 각 폴더별 가이드 작성

**생성할 README 목록**:

1. `docs/README.md` - 전체 문서 구조 설명
2. `docs/01_citations/README.md` - 인용 자료 가이드
3. `docs/02_research/README.md` - 조사 자료 가이드
4. `docs/03_ai_references/README.md` - AI 레퍼런스 가이드
5. `docs/04_project_docs/README.md` - 프로젝트 문서 가이드
6. `thesis/README.md` - 논문 작성 가이드

**체크리스트**:

- [ ] docs/README.md 작성
- [ ] 각 하위 폴더 README.md 작성
- [ ] thesis/README.md 업데이트
- [ ] 메인 README.md 문서 구조 섹션 업데이트

---

### Step 5: 링크 업데이트 (Link Updates)

**기간**: 0.5일
**목표**: 모든 문서 간 링크 수정

```bash
# 링크 참조 검색
grep -r "\[.*\](.*\.md)" docs/ thesis/

# 깨진 링크 자동 탐지 스크립트 실행
python scripts/check_broken_links.py
```

**체크리스트**:

- [ ] 문서 내부 링크 수정
- [ ] README 파일 링크 검증
- [ ] 상대 경로 확인
- [ ] 깨진 링크 수정

---

### Step 6: 구조 정리 (Cleanup)

**기간**: 0.5일
**목표**: 기존 폴더 제거 및 정리

```bash
# 빈 폴더 제거
rmdir claudedocs/references/perplexity
rmdir claudedocs/references/arxiv
rmdir claudedocs/references
rmdir claudedocs

rmdir thesis/references
rmdir thesis/draft
```

**체크리스트**:

- [ ] claudedocs/ 폴더 제거
- [ ] thesis/references/ 폴더 제거
- [ ] .gitignore 업데이트
- [ ] 불필요한 파일 삭제

---

### Step 7: 검증 및 테스트 (Validation)

**기간**: 0.5일
**목표**: 문서 구조 검증

```bash
# 문서 트리 확인
tree docs/ thesis/ -L 3

# 모든 마크다운 파일 렌더링 테스트
find docs/ thesis/ -name "*.md" -exec markdown-lint {} \;

# 링크 검증
python scripts/validate_documentation.py
```

**체크리스트**:

- [ ] 디렉토리 구조 검증
- [ ] 마크다운 문법 검증
- [ ] 링크 유효성 검증
- [ ] 이미지 경로 검증

---

### Step 8: 커밋 및 PR (Commit & PR)

**기간**: 0.5일
**목표**: 변경사항 커밋 및 리뷰

```bash
# 변경사항 커밋
git add .
git commit -m "docs: refactor documentation structure for better organization

- Separate citations, research, AI references, and project docs
- Reorganize thesis manuscript into clear sections
- Remove duplicate reference folders
- Add comprehensive README files for each category
- Update all internal links

BREAKING CHANGE: Documentation paths have changed
See DOCUMENTATION_REFACTORING_PLAN.md for migration guide"

# PR 생성
git push origin refactor/documentation-structure
```

**체크리스트**:

- [ ] Git 커밋 메시지 작성
- [ ] PR 생성 및 설명 작성
- [ ] 리뷰 요청
- [ ] CI/CD 검증 통과

---

## 📊 진행 상황 추적 (Progress Tracking)

### Overall Progress

- [ ] **Step 1**: 준비 단계 (0/4 tasks)
- [ ] **Step 2**: 디렉토리 구조 생성 (0/6 tasks)
- [ ] **Step 3**: 문서 이동 (0/5 tasks)
- [ ] **Step 4**: README 파일 생성 (0/4 tasks)
- [ ] **Step 5**: 링크 업데이트 (0/4 tasks)
- [ ] **Step 6**: 구조 정리 (0/4 tasks)
- [ ] **Step 7**: 검증 및 테스트 (0/4 tasks)
- [ ] **Step 8**: 커밋 및 PR (0/4 tasks)

**총 진행률**: 0/35 tasks (0%)

---

## 🎓 기대 효과 (Expected Benefits)

### 1. 명확한 문서 분류

- ✅ 인용 → 조사 → AI 레퍼런스 → 프로젝트 문서로 명확한 흐름
- ✅ 각 문서의 목적과 출처를 쉽게 파악

### 2. 개선된 검색성

- ✅ 폴더 구조만으로도 원하는 문서 위치 예측 가능
- ✅ 일관된 네이밍 규칙으로 파일 검색 용이

### 3. 논문 작성 효율성

- ✅ 논문 섹션별로 문서 분리되어 작성 편의성 증가
- ✅ 참고 자료 위치가 명확해 인용 작업 간소화

### 4. 협업 용이성

- ✅ 새로운 팀원도 문서 구조 빠르게 이해
- ✅ 각 폴더별 README로 자가 학습 가능

### 5. 유지보수성

- ✅ 단일 진실 공급원 (중복 제거)
- ✅ 명확한 파일 이동 규칙으로 미래 확장 용이

---

## 🔗 관련 문서 (Related Documents)

- [README.md](../README.md) - 프로젝트 메인 문서
- [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md) - 기존 마이그레이션 가이드
- `.gitignore` - 문서 제외 규칙

---

## 📞 문의 및 피드백 (Contact & Feedback)

**작성자**: Claude + 사용자
**날짜**: 2025-10-27
**버전**: 1.0.0

**질문 및 제안**:

- GitHub Issues에 등록
- PR 리뷰 코멘트로 피드백

---

## 📜 변경 이력 (Change Log)

### v1.0.0 (2025-10-27)

- 초기 리팩토링 계획 수립
- 4개 주요 카테고리 정의 (citations, research, ai_references, project_docs)
- 마이그레이션 매핑 완료
- 8단계 실행 계획 작성
