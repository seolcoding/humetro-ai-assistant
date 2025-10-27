# 📚 Research Documentation

**Purpose**: 모든 연구 관련 문서의 중앙 저장소
**Last Updated**: 2025-10-27

---

## 📂 Directory Structure

```
docs/
├── 01_citations/          # 🎓 학술 인용 자료
├── 02_research/           # 🔍 조사 및 분석 자료
├── 03_ai_references/      # 🤖 AI 생성 종합 자료
└── 04_project_docs/       # 📋 프로젝트 문서
```

---

## 🎓 01_citations/ - 학술 인용 자료

### 목적
학술 논문, arXiv 논문 요약, 서베이 논문 등 **인용 가능한 학술 자료**

### 구조
- `arxiv_papers/` - arXiv 논문 요약 및 분석
  - `graph_rag/` - Graph RAG 관련 논문
  - `evaluation/` - 평가 메트릭 관련 논문
  - `llm_deployment/` - LLM 배포 관련 논문
- `academic_surveys/` - 학술 서베이 논문 정리

### 사용 시기
- 논문 Related Work 섹션 작성 시
- 기술적 근거가 필요할 때
- 최신 연구 동향 파악 시

**📖 상세 가이드**: [01_citations/README.md](./01_citations/README.md)

---

## 🔍 02_research/ - 조사 및 분석 자료

### 목적
Perplexity, 웹 검색 등을 통한 **비학술적 조사 자료 및 분석**

### 구조
- `perplexity_deep_research/` - Perplexity Academic 모드 심층 조사
- `web_research/` - 웹 검색 기반 조사
- `competitive_analysis/` - 경쟁 기술/프레임워크 분석

### 사용 시기
- 빠른 기술 조사가 필요할 때
- 구현 방법 탐색 시
- 최신 트렌드 파악 시

**📖 상세 가이드**: [02_research/README.md](./02_research/README.md) (Coming Soon)

---

## 🤖 03_ai_references/ - AI 생성 종합 자료

### 목적
Claude 등 AI가 생성한 **종합 가이드 및 레퍼런스 자료**

### 주요 문서
- `claude_comprehensive_guide.md` - Claude 활용 RAG/KG 연구 종합 가이드
- `mcp_servers_catalog.md` - MCP 서버 카탈로그 (예정)
- `vector_db_comparison.md` - 벡터 DB 비교 분석 (예정)
- `implementation_patterns.md` - 구현 패턴 모음 (예정)

### 사용 시기
- 프로젝트 초기 설계 시
- 기술 스택 선정 시
- 구현 패턴 참고 시

**📖 상세 가이드**: [03_ai_references/README.md](./03_ai_references/README.md) (Coming Soon)

---

## 📋 04_project_docs/ - 프로젝트 문서

### 목적
프로젝트 아키텍처, 가이드, 명세서 등 **실무 문서**

### 구조
- `architecture/` - 시스템 설계 및 아키텍처 문서
- `guides/` - 개발, 배포, 평가 가이드
- `specifications/` - 상세 명세서

### 사용 시기
- 프로젝트 구현 시
- 팀 온보딩 시
- 시스템 이해 필요 시

**📖 상세 가이드**: [04_project_docs/README.md](./04_project_docs/README.md)

---

## 🧭 Quick Navigation

### 내가 찾는 문서는?

| 질문 | 위치 | 예시 |
|-----|------|-----|
| 논문에 인용할 학술 자료는? | `01_citations/` | arXiv 논문 요약 |
| 빠른 기술 조사 자료는? | `02_research/` | Perplexity 심층 조사 |
| AI가 만든 종합 가이드는? | `03_ai_references/` | Claude 종합 가이드 |
| 프로젝트 구현 문서는? | `04_project_docs/` | 시스템 설계, 가이드 |

### 시나리오별 가이드

**시나리오 1: 논문 Related Work 작성**
```
1. docs/01_citations/arxiv_papers/ 에서 관련 논문 확인
2. docs/02_research/perplexity_deep_research/ 에서 최신 동향 파악
3. 인용 자료 선정 후 논문 작성
```

**시나리오 2: 새로운 기술 스택 선정**
```
1. docs/03_ai_references/claude_comprehensive_guide.md 에서 전체 옵션 확인
2. docs/02_research/competitive_analysis/ 에서 비교 분석
3. docs/04_project_docs/architecture/ 에 결정 사항 문서화
```

**시나리오 3: 시스템 구현**
```
1. docs/04_project_docs/architecture/ 에서 설계 확인
2. docs/04_project_docs/guides/ 에서 구현 가이드 참조
3. docs/01_citations/ 에서 기술적 근거 확인
```

---

## 📝 Documentation Best Practices

### 1. 새 문서 추가 시
- 적절한 카테고리 폴더에 배치
- 일관된 네이밍 규칙 사용 (snake_case.md)
- 파일 상단에 메타데이터 추가 (작성일, 목적 등)

### 2. 기존 문서 수정 시
- 변경 이력 하단에 기록
- 주요 변경 시 버전 번호 업데이트
- 관련 링크도 함께 업데이트

### 3. 삭제 전 확인
- 다른 문서에서 참조 여부 확인
- 중요 문서는 archive/ 폴더로 이동 고려

---

## 🔗 Related Resources

- [../README.md](../README.md) - 프로젝트 메인 문서
- [../thesis/README.md](../thesis/README.md) - 논문 작성 가이드
- [DOCUMENTATION_REFACTORING_PLAN.md](../DOCUMENTATION_REFACTORING_PLAN.md) - 문서 구조 변경 계획

---

**Last Updated**: 2025-10-27
**Maintainer**: Research Team
