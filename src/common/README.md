# Common Utilities

공통 유틸리티 모듈로 프로젝트 전체에서 사용되는 기능들을 제공합니다.

## 모듈

### `logger.py` - 중앙 로깅 시스템

RAG 파이프라인을 위한 종합적인 로깅 시스템을 제공합니다.

**주요 기능:**
- 다중 로그 파일 (pipeline, API calls, errors, metrics)
- 구조화된 JSON 로깅
- API 호출 비용 추적
- 실험별 로그 디렉토리

**사용 예시:**
```python
from src.common.logger import get_logger

logger = get_logger("my_experiment")

# 기본 로깅
logger.info("Pipeline started")
logger.error("An error occurred")

# API 호출 로깅
logger.log_api_call(
    provider="openai",
    model="text-embedding-3-small",
    operation="embedding",
    tokens=100,
    cost=0.002,
    cached=False
)

# 메트릭 로깅
logger.log_metric("accuracy", 0.95, metadata={"model": "gpt-4"})

# 스테이지 로깅
logger.log_stage_start(1, "Data Collection")
# ... 작업 수행 ...
logger.log_stage_end(1, "Data Collection", duration_seconds=10.5)

# 통계 출력
logger.print_summary()
```

### `env_loader.py` - 환경변수 관리

.env 파일에서 환경변수를 로드하고 검증하는 유틸리티를 제공합니다.

**주요 기능:**
- .env 파일 로딩
- 필수 변수 검증
- 타입 변환 (int, bool, Path)
- RAG 파이프라인 전용 환경변수 로더

**사용 예시:**
```python
from src.common.env_loader import EnvLoader, load_rag_env

# RAG 환경변수 로드
load_rag_env(require_neo4j=True, require_ollama=False)

# 개별 변수 접근
api_key = EnvLoader.get_required_env("OPENAI_API_KEY")
data_dir = EnvLoader.get_env_path("DATA_DIR")
chunk_size = EnvLoader.get_env_int("CHUNK_SIZE", default=1024)
use_cache = EnvLoader.get_env_bool("USE_CACHE", default=True)

# 다중 변수 검증
values = EnvLoader.validate_env_vars(["VAR1", "VAR2", "VAR3"])
```

### `project_paths.py` - 프로젝트 경로 유틸리티

프로젝트 루트 및 주요 디렉토리 경로를 관리합니다 (기존 모듈).

**사용 예시:**
```python
from src.common.project_paths import get_project_root

project_root = get_project_root()
data_dir = project_root / "data"
results_dir = project_root / "results"
```

## 테스트

```bash
# 모든 테스트 실행
pytest tests/test_common/ -v

# 개별 모듈 테스트
pytest tests/test_common/test_logger.py -v
pytest tests/test_common/test_env_loader.py -v
```

## 의존성

- `python-dotenv` - 환경변수 로딩
- `logging` - 표준 로깅
- `pathlib` - 경로 처리

## 필수 환경변수

`.env` 파일에 다음 변수들을 설정하세요:

```env
# 필수
OPENAI_API_KEY=sk-your-api-key
DATA_DIR=./data
RESULTS_DIR=./results

# 선택 (Knowledge Graph 사용 시)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# 선택 (Ollama 사용 시)
OLLAMA_BASE_URL=http://localhost:11434
```
