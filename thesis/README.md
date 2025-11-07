# 📝 Thesis Writing Area

**Title**: 온프레미스 오픈소스 기반 Graph RAG 시스템의 공공부문 적용 연구
**Author**: [Your Name]
**Last Updated**: 2025-10-27

---

## 📂 Directory Structure

```
thesis/
├── manuscript/         # 📄 논문 원고 (섹션별)
├── figures/           # 📊 그림 파일
├── tables/            # 📈 표 데이터
└── drafts/            # 💾 초안 및 버전 관리
```

---

## 📄 manuscript/ - 논문 원고

### 섹션별 파일

| 파일 | 섹션 | 상태 |
|-----|------|-----|
| `00_outline.md` | 전체 개요 | ✅ 초안 |
| `01_introduction.md` | 서론 | 📝 작성 예정 |
| `02_related_work.md` | 관련 연구 | 📝 작성 예정 |
| `03_methodology.md` | 연구 방법 | 📝 작성 예정 |
| `04_experiments.md` | 실험 및 평가 | 📝 작성 예정 |
| `05_results.md` | 결과 및 분석 | 📝 작성 예정 |
| `06_discussion.md` | 논의 | 📝 작성 예정 |
| `07_conclusion.md` | 결론 | 📝 작성 예정 |
| `08_references.bib` | 참고문헌 (BibTeX) | 📝 작성 예정 |

### 작성 순서 권장
1. `00_outline.md` - 전체 구조 확립
2. `02_related_work.md` - 선행 연구 정리
3. `03_methodology.md` - 연구 방법 설계
4. `04_experiments.md` - 실험 계획
5. `01_introduction.md` - 서론 작성
6. `05_results.md` - 결과 정리
7. `06_discussion.md` - 논의 작성
8. `07_conclusion.md` - 결론 정리

---

## 📊 figures/ - 그림 파일

### 네이밍 규칙
```
[장]_[섹션]_[설명].png

예시:
03_01_rag_pipeline_architecture.png
04_02_performance_comparison_chart.png
05_01_cost_analysis_graph.png
```

### 요구사항
- 형식: PNG (300 DPI 이상)
- 크기: 최대 폭 15cm
- 파일명: 영문 소문자, 언더스코어 사용

---

## 📈 tables/ - 표 데이터

### 네이밍 규칙
```
[장]_[섹션]_[설명].csv

예시:
02_01_related_work_comparison.csv
04_01_experimental_setup.csv
05_01_performance_results.csv
```

### 형식
- CSV 형식 (UTF-8 인코딩)
- 첫 행은 헤더
- 수치 데이터는 통일된 소수점 자릿수

---

## 💾 drafts/ - 초안 관리

### 버전 관리
```
v1_[섹션명].md  - 초안
v2_[섹션명].md  - 지도교수 리뷰 반영
v3_[섹션명].md  - 최종 수정

예시:
v1_introduction.md
v2_introduction_advisor_review.md
final_submission/
```

---

## 🎯 Writing Guidelines

### 1. 학술 논문 작성 원칙
- 객관적이고 명확한 문체
- 근거 기반 주장 (인용 필수)
- 논리적 흐름 유지
- 재현 가능한 방법론 제시

### 2. 인용 규칙
- APA 또는 IEEE 스타일 통일
- [../docs/01_citations/](../docs/01_citations/) 활용
- 직접 인용 시 페이지 번호 명시

### 3. 그림/표 작성
- 캡션 필수 (Figure 1. ~, Table 1. ~)
- 본문에서 반드시 언급
- 독립적으로 이해 가능하도록 설명

---

## 📚 Reference Resources

### 논문 작성 참고
- [../docs/01_citations/](../docs/01_citations/) - 인용 가능한 학술 자료
- [../docs/02_research/](../docs/02_research/) - 배경 조사 자료
- [../docs/04_project_docs/](../docs/04_project_docs/) - 프로젝트 상세 문서

### 관련 문서
- [00_outline.md](./manuscript/00_outline.md) - 논문 전체 개요
- [../README.md](../README.md) - 프로젝트 메인 문서

---

## ✅ Submission Checklist

### 제출 전 확인사항
- [ ] 모든 섹션 작성 완료
- [ ] 그림/표 캡션 확인
- [ ] 참고문헌 형식 통일
- [ ] 오탈자 검토
- [ ] 지도교수 최종 승인
- [ ] PDF 변환 확인
- [ ] 파일명 규격 확인

---

**Advisor**: [Advisor Name]
**Institution**: [University Name]
**Expected Completion**: 2025-XX-XX
