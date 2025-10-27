# 📋 Migration Guide - 프로젝트 구조 전환 가이드

## 🎯 전환 목표

Frontend 중심의 프로토타입에서 **논문 연구용 실험 프레임워크**로 전환

## 📁 주요 변경사항

### 제거된 파일들
```
✗ main.py (Streamlit app)
✗ src/legacy/app.py
✗ assets/ (frontend resources)
✗ 기타 프론트엔드 관련 파일
```
→ `archive/` 폴더로 이동 (필요시 참조 가능)

### 새로운 구조
```
✓ data/               # 체계적인 데이터 관리
✓ experiments/        # 실험 설정 및 추적
✓ results/            # 자동화된 결과 저장
✓ scripts/            # 실험 자동화 스크립트
✓ thesis/             # 논문 자료 집중화
```

## 🔧 데이터 이동 매핑

| 이전 위치 | 새 위치 | 설명 |
|-----------|---------|------|
| `datasets/` | `data/raw/` | 원본 데이터셋 |
| `crawl_result/` | `data/raw/crawled/` | 크롤링 데이터 |
| `vectorstore/` | `data/vectorstore/` | 벡터 저장소 |
| `src/llm_tools/` | `src/rag_pipeline/tools/` | LLM 도구 |
| `src/data_processor/` | `src/data_processing/` | 데이터 처리 |
| `thesis_docs/` | `thesis/draft/` | 논문 초안 |
| `*.md` | `thesis/references/` | 참고 문서 |

## 💻 코드 임포트 변경

### 이전
```python
from src.llm_tools.HumetroFare import FareCalculator
from src.data_processor.evaluate_rag import evaluate
```

### 이후
```python
from src.rag_pipeline.tools.HumetroFare import FareCalculator
from src.evaluation.evaluator import RAGASEvaluator
```

## 🚀 실험 실행 방법

### 1. 데이터 준비
```bash
# 다산콜센터 FAQ 다운로드
python scripts/download_data.py --source dasan_faq

# 지식 그래프 구축 (일회성)
python scripts/build_knowledge_graph.py --model gpt-5
```

### 2. 실험 실행
```bash
# 전체 16개 시스템 실험
python scripts/run_experiment.py

# 특정 조합만 실행
python scripts/run_experiment.py \
    --models gemma_3_12b \
    --methods graph_rag
```

### 3. 결과 분석
```bash
# 결과 분석 및 시각화
python scripts/analyze_results.py \
    --results-dir results/latest

# 논문용 표 생성
python scripts/analyze_results.py \
    --format tables
```

## ⚠️ 주의사항

1. **데이터 백업**: 마이그레이션 전 중요 데이터 백업
2. **환경 변수**: `.env` 파일 확인 및 업데이트
3. **의존성**: `requirements.txt` 재설치 필요
4. **Git 브랜치**: `thesis` 브랜치에서 작업

## 📊 연구 워크플로우

```mermaid
graph LR
    A[데이터 수집] --> B[전처리]
    B --> C[지식그래프 구축]
    C --> D[RAG 파이프라인]
    D --> E[실험 실행]
    E --> F[평가]
    F --> G[결과 분석]
    G --> H[논문 작성]
```

## 🔄 롤백 방법

필요시 이전 구조로 복원:
```bash
# archive에서 복원
cp -r archive/frontend/main.py .
cp -r archive/legacy/* src/legacy/

# 이전 커밋으로 복원
git checkout main
```

## 📝 체크리스트

- [ ] 데이터 백업 완료
- [ ] archive 폴더 생성 및 이동
- [ ] 새 디렉토리 구조 생성
- [ ] 파일 이동 완료
- [ ] 코드 임포트 경로 수정
- [ ] .gitignore 업데이트
- [ ] README 업데이트
- [ ] 테스트 실행 확인

## 🆘 문제 해결

### Q: 임포트 에러 발생
A: `sys.path`에 프로젝트 루트 추가
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
```

### Q: 데이터 파일을 찾을 수 없음
A: 새 경로 확인
```bash
find . -name "*.csv" -o -name "*.json"
```

### Q: 실험 스크립트 실행 안됨
A: 실행 권한 부여
```bash
chmod +x scripts/*.py
```

---

> **Note**: 이 가이드는 thesis 브랜치 기준입니다. main 브랜치는 안정 버전을 유지합니다.