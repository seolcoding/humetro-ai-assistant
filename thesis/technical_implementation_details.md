# 기술적 구현 세부사항: RAG 파이프라인 개발 과정

## 1. 개발 환경 및 프로젝트 구조

### 1.1 개발 환경 설정

본 연구의 개발 환경은 다음과 같이 구성되었다:

- **운영체제**: macOS Darwin 24.6.0
- **Python 버전**: 3.12 (UV 패키지 매니저를 통해 고정)
- **주요 의존성 관리**: `pyproject.toml` 및 `uv.lock` 파일 활용
- **버전 관리**: Git (feature branch 전략 적용)

### 1.2 프로젝트 디렉토리 구조

```
humetro-ai-assistant/
├── src/
│   ├── rag_pipeline/
│   │   ├── stages/                 # 6단계 파이프라인 모듈
│   │   │   ├── stage_01_data_collection.py
│   │   │   ├── stage_02_chunking.py
│   │   │   ├── stage_03_embedding.py
│   │   │   ├── stage_05_vector_store.py
│   │   │   └── stage_06_retrieval.py
│   │   ├── testset_generator.py    # 테스트셋 생성 시스템
│   │   └── generate_benchmark_50q.py # 벤치마크 생성 CLI
│   └── crawler/                    # 웹 크롤링 모듈
├── data/
│   ├── crawled/                    # 크롤링된 원시 데이터
│   │   └── seoul_traffic/
│   │       └── markdown_deduplicated/
│   └── evaluation/                 # 평가 관련 데이터
│       ├── testsets/               # 생성된 테스트셋
│       │   ├── metadata.json       # 버전 관리 메타데이터
│       │   └── testset_*.{json,csv,md}
│       └── llm_comparison/         # LLM 비교 결과
├── tests/                          # 테스트 코드
│   └── test_rag_pipeline/
├── thesis/                         # 논문 관련 문서
└── claudedocs/                     # 프로젝트 문서화
```

## 2. 데이터 수집 및 전처리 구현

### 2.1 크롤링 시스템 (Crawl4AI 활용)

웹 크롤링은 Crawl4AI 프레임워크를 활용하여 구현하였다. 서울시 교통정보 포털의 특성을 고려한 맞춤형 크롤러를 개발하였으며, 다음과 같은 기능을 구현하였다:

```python
class SeoulTrafficCrawler:
    def __init__(self):
        self.base_url = "https://topis.seoul.go.kr"
        self.session_config = {
            "timeout": 30,
            "max_retries": 3,
            "wait_time": 1  # 서버 부하 방지
        }

    async def crawl_with_markdown(self, url: str) -> str:
        """마크다운 형식으로 페이지 내용 추출"""
        # JavaScript 렌더링 대기
        await page.wait_for_selector('.content-area')
        # 마크다운 변환
        content = html2markdown(html_content)
        return content
```

### 2.2 중복 제거 알고리즘

SHA-256 해시를 활용한 효율적인 중복 제거 시스템을 구현하였다:

```python
def deduplicate_documents(documents: List[Document]) -> List[Document]:
    """문서 중복 제거"""
    seen_hashes = set()
    unique_docs = []

    for doc in documents:
        # 내용 기반 해시 생성
        content_hash = hashlib.sha256(
            doc.page_content.encode('utf-8')
        ).hexdigest()

        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)

    # 통계 저장
    stats = {
        "original_count": len(documents),
        "unique_count": len(unique_docs),
        "duplicates_removed": len(documents) - len(unique_docs),
        "timestamp": datetime.now().isoformat()
    }

    with open("deduplication_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return unique_docs
```

## 3. RAG 파이프라인 단계별 구현

### 3.1 Stage 1: 데이터 수집 구현

데이터 수집 단계는 캐시 확인, 문서 로드, 마크다운 정제의 3단계로 구성된다:

```python
class DataCollectionStage:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cleaning_patterns = self._compile_cleaning_patterns()

    def _compile_cleaning_patterns(self) -> List[re.Pattern]:
        """정제용 정규표현식 패턴 컴파일"""
        patterns = [
            r'\[인쇄\]\([^)]+\)',
            r'\[\s*카카오톡\s*\]\([^)]+\)',
            r'\[\*\*네이버blog\*\*\]\([^)]+\)',
            r'\[\s*페이스북\s*\]\([^)]+\)',
            r'\[\s*트위터\s*\]\([^)]+\)',
            r'\[\s*메일전송\s*\]\([^)]+\)',
            r'\[\s*스크랩\s*\]\([^)]+\)',
            r'\[\s*소스복사\s*\]\([^)]+\)',
        ]
        return [re.compile(p) for p in patterns]

    def clean_markdown(self, content: str) -> str:
        """마크다운 문서 정제"""
        cleaned = content

        # UI 요소 제거
        for pattern in self.cleaning_patterns:
            cleaned = pattern.sub('', cleaned)

        # 과도한 줄바꿈 정규화 (최대 2개)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        return cleaned.strip()
```

### 3.2 Stage 2: 청킹 전략

한국어 텍스트 특성을 고려한 청킹 전략을 구현하였다:

```python
class ChunkingStage:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "。", "!", "?", " ", ""],
            length_function=self._korean_aware_length,
        )

    def _korean_aware_length(self, text: str) -> int:
        """한국어 문자 길이 계산 (한글 2바이트 고려)"""
        korean_chars = len([c for c in text if '가' <= c <= '힣'])
        other_chars = len(text) - korean_chars
        # 한글은 평균 1.5배 가중치 적용
        return int(korean_chars * 1.5 + other_chars)

    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """문서를 청크로 분할"""
        all_chunks = []

        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            # 메타데이터 보존 및 청크 인덱스 추가
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_index'] = i
                chunk.metadata['source_doc_id'] = doc.metadata.get('doc_id')
            all_chunks.extend(chunks)

        return all_chunks
```

### 3.3 Stage 3: 임베딩 최적화

API 호출 최적화를 위한 배치 처리 시스템을 구현하였다:

```python
class EmbeddingStage:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            chunk_size=100,  # 배치 크기
            max_retries=3,
            request_timeout=30
        )
        self.dimension = 768  # text-embedding-3-small 차원

    async def generate_embeddings_batch(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """배치 단위 임베딩 생성"""
        embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # 재시도 로직 포함
            for attempt in range(3):
                try:
                    batch_embeddings = await self.embeddings.aembed_documents(batch)
                    embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 ** attempt)  # 지수 백오프

        return embeddings
```

### 3.4 Stage 5: 벡터 스토어 구현

FAISS를 활용한 Cosine Similarity 기반 벡터 검색 시스템을 구현하였다:

```python
class VectorStoreStage:
    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """Cosine Similarity를 사용하는 FAISS 벡터 스토어 생성"""
        # 임베딩 생성
        texts = [doc.page_content for doc in chunks]
        embeddings_list = self.embeddings.embed_documents(texts)
        embeddings_array = np.array(embeddings_list, dtype='float32')

        # Cosine Similarity를 위한 벡터 정규화
        faiss.normalize_L2(embeddings_array)

        # Inner Product 인덱스 생성 (정규화된 벡터에서 IP = Cosine Similarity)
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_array)

        # FAISS 벡터 스토어 생성
        vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            normalize_L2=True  # 쿼리 벡터도 자동 정규화
        )

    def search_with_mmr(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """MMR (Maximal Marginal Relevance) 검색"""
        # 더 많은 후보를 가져온 후 다양성 기반 선택
        return self.vector_store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult  # 관련성 vs 다양성 균형
        )
```

## 4. 테스트셋 생성 시스템

### 4.1 RAGAS 통합 및 한국어 최적화

RAGAS TestSet Generator를 한국어 환경에 최적화하여 통합하였다:

```python
class CachedTestsetGenerator:
    def __init__(self):
        self.cache_dir = Path("data/evaluation/testsets")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _create_korean_personas(self) -> List[Persona]:
        """한국어 전용 페르소나 생성"""
        from ragas.testset.persona import Persona

        personas = [
            Persona(
                name="서울 시민",
                role_description=(
                    "서울에 거주하는 한국인으로, 서울 대중교통 정보를 "
                    "한국어로 질문합니다. 일상적인 통근/통학 경로와 "
                    "교통 상황에 관심이 많습니다."
                )
            ),
            Persona(
                name="외국인 거주자",
                role_description=(
                    "서울에 거주하는 외국인으로, 서울 대중교통 정보를 "
                    "한국어로 질문합니다. 한국어를 배우고 있으며 "
                    "간단명료한 한국어 답변을 선호합니다."
                )
            ),
            Persona(
                name="교통 관심 시민",
                role_description=(
                    "서울 교통 정책과 노선 변경에 관심이 많은 한국인입니다. "
                    "한국어로 상세한 교통 정보를 질문하고 구체적인 답변을 원합니다."
                )
            ),
        ]

        return personas

    def generate_testset(
        self,
        documents: List[Document],
        testset_size: int = 50,
        use_korean_personas: bool = True
    ) -> pd.DataFrame:
        """테스트셋 생성 (캐싱 포함)"""
        # 캐시 키 생성
        cache_key = self._generate_cache_key(documents, testset_size)

        # 캐시 확인
        cached = self._check_cache(cache_key)
        if cached:
            logger.info(f"💾 캐시에서 로드: {cache_key}")
            return cached['testset_df']

        # 새로 생성
        logger.info("🔄 새 테스트셋 생성 중...")

        # RAGAS Generator 초기화
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        generator_kwargs = {
            "llm": llm,
            "embedding_model": embeddings,
        }

        # 한국어 페르소나 적용
        if use_korean_personas:
            generator_kwargs["persona_list"] = self._create_korean_personas()

        generator = TestsetGenerator(**generator_kwargs)

        # 테스트셋 생성
        testset = generator.generate_with_langchain_docs(
            documents=documents[:testset_size],  # 문서 수 제한
            testset_size=testset_size,
            with_debugging_logs=True
        )

        # DataFrame 변환 및 캐시 저장
        testset_df = testset.to_pandas()
        self._save_cache(cache_key, testset_df)

        return testset_df
```

### 4.2 캐싱 메커니즘

효율적인 캐싱을 위한 해시 기반 시스템:

```python
def _generate_cache_key(
    self,
    documents: List[Document],
    testset_size: int
) -> str:
    """결정적 캐시 키 생성"""
    # 설정 정보를 JSON으로 직렬화
    config = {
        "doc_count": len(documents),
        "testset_size": testset_size,
        "doc_hashes": [
            hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
            for doc in documents[:5]  # 처음 5개 문서만 해시
        ],
        "model": "gpt-4o-mini",
        "temperature": 0.3,
    }

    # SHA-256 해시 생성
    config_str = json.dumps(config, sort_keys=True)
    full_hash = hashlib.sha256(config_str.encode()).hexdigest()

    # 16자리로 축약
    return full_hash[:16]

def _save_cache(
    self,
    cache_key: str,
    testset_df: pd.DataFrame
):
    """캐시 저장 (JSON, CSV, Markdown)"""
    base_path = self.cache_dir / f"testset_{cache_key}"

    # JSON 저장
    json_data = {
        "cache_key": cache_key,
        "created_at": datetime.now().isoformat(),
        "testset": testset_df.to_dict(orient='records'),
        "stats": {
            "num_questions": len(testset_df),
            "avg_context_length": testset_df['contexts'].str.len().mean(),
            "avg_answer_length": testset_df['ground_truth'].str.len().mean(),
        }
    }

    with open(f"{base_path}.json", 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    # CSV 저장
    testset_df.to_csv(f"{base_path}.csv", index=False, encoding='utf-8')

    # Markdown 문서 생성
    self._generate_markdown_doc(cache_key, testset_df)
```

## 5. 평가 시스템 구현

### 5.1 RAGAS 평가 파이프라인

다중 모델 평가를 위한 자동화된 파이프라인:

```python
class RAGASEvaluator:
    def __init__(self, judge_model: str = "gpt-5"):
        self.judge_llm = self._create_judge_llm(judge_model)
        self.metrics = [
            faithfulness,
            answer_relevancy,
            answer_correctness,
        ]

    def _create_judge_llm(self, model_name: str):
        """평가자 LLM 생성 (GPT-5 특별 처리)"""
        if model_name == "gpt-5":
            # GPT-5는 temperature=1만 지원
            return ChatOpenAI(
                model="gpt-5",
                temperature=None,  # 패치된 RAGAS가 처리
                max_tokens=4096
            )
        else:
            return ChatOpenAI(
                model=model_name,
                temperature=0.0,
                max_tokens=4096
            )

    async def evaluate_model(
        self,
        model_name: str,
        testset_df: pd.DataFrame,
        rag_chain: Any
    ) -> Dict[str, float]:
        """단일 모델 평가"""
        # RAG 체인으로 답변 생성
        responses = []
        contexts_list = []

        for _, row in testset_df.iterrows():
            question = row['question']

            # RAG 체인 실행
            result = await rag_chain.ainvoke({"question": question})
            responses.append(result['answer'])
            contexts_list.append(result['contexts'])

        # 평가 데이터셋 구성
        eval_dataset = Dataset.from_dict({
            'question': testset_df['question'].tolist(),
            'answer': responses,
            'contexts': contexts_list,
            'ground_truth': testset_df['ground_truth'].tolist()
        })

        # RAGAS 평가 실행
        results = evaluate(
            dataset=eval_dataset,
            metrics=self.metrics,
            llm=self.judge_llm,
        )

        return {
            "model": model_name,
            "faithfulness": results['faithfulness'],
            "answer_relevancy": results['answer_relevancy'],
            "answer_correctness": results['answer_correctness'],
            "timestamp": datetime.now().isoformat()
        }
```

### 5.2 LiteLLM을 통한 다중 모델 지원

다양한 LLM 제공자를 통합 관리:

```python
class MultiModelEvaluator:
    def __init__(self):
        self.models = {
            "gpt-4o-mini": {
                "provider": "openai",
                "config": {"temperature": 0.3}
            },
            "exaone-3.5-7.8b": {
                "provider": "openrouter",
                "config": {"temperature": 0.3, "max_tokens": 2048}
            },
            "qwen3-8b": {
                "provider": "openrouter",
                "config": {"temperature": 0.3}
            },
            "gemma3-12b": {
                "provider": "openrouter",
                "config": {"temperature": 0.3}
            },
            "gpt-oss-20b": {
                "provider": "openrouter",
                "config": {"temperature": 0.3, "max_tokens": 4096}
            }
        }

    def create_llm(self, model_name: str):
        """LiteLLM을 통한 통합 LLM 생성"""
        model_config = self.models[model_name]

        if model_config["provider"] == "openai":
            return ChatOpenAI(
                model=model_name,
                **model_config["config"]
            )
        elif model_config["provider"] == "openrouter":
            return ChatLiteLLM(
                model=f"openrouter/{model_name}",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                **model_config["config"]
            )
```

## 6. 메타데이터 관리 시스템

### 6.1 버전 관리 구현

테스트셋 버전을 체계적으로 관리하는 시스템:

```python
class MetadataManager:
    def __init__(self, metadata_file: Path):
        self.metadata_file = metadata_file
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """메타데이터 파일 로드"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "latest": None,
                "benchmark": None,
                "versions": []
            }

    def register_version(
        self,
        cache_key: str,
        testset_info: Dict,
        is_latest: bool = False,
        is_benchmark: bool = False
    ):
        """새 버전 등록"""
        # 기존 latest 플래그 해제
        if is_latest:
            for version in self.metadata["versions"]:
                version["is_latest"] = False
            self.metadata["latest"] = cache_key

        # 벤치마크 플래그 설정
        if is_benchmark:
            for version in self.metadata["versions"]:
                version["is_benchmark"] = False
            self.metadata["benchmark"] = cache_key

        # 새 버전 추가
        version_entry = {
            "cache_key": cache_key,
            "created_at": datetime.now().isoformat(),
            **testset_info,
            "is_latest": is_latest,
            "is_benchmark": is_benchmark
        }

        self.metadata["versions"].append(version_entry)
        self._save_metadata()

    def get_latest_version(self) -> Optional[str]:
        """최신 버전 캐시 키 반환"""
        return self.metadata.get("latest")

    def get_benchmark_version(self) -> Optional[str]:
        """벤치마크 버전 캐시 키 반환"""
        return self.metadata.get("benchmark")
```

## 7. 성능 최적화 및 모니터링

### 7.1 프로그레스 추적

실시간 진행 상황 모니터링:

```python
from tqdm import tqdm
import logging

class ProgressTracker:
    def __init__(self, total_steps: int, description: str):
        self.pbar = tqdm(
            total=total_steps,
            description=description,
            unit="step",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        self.logger = logging.getLogger(__name__)

    def update(self, message: str = ""):
        """진행 상황 업데이트"""
        self.pbar.update(1)
        if message:
            self.pbar.set_postfix_str(message)
            self.logger.info(message)

    def close(self):
        """프로그레스 바 종료"""
        self.pbar.close()
```

### 7.2 에러 처리 및 재시도

안정적인 실행을 위한 에러 처리:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustExecutor:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    async def execute_with_retry(self, func, *args, **kwargs):
        """재시도 로직을 포함한 실행"""
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"실행 실패: {e}")
            raise
```

## 8. 테스트 커버리지

### 8.1 단위 테스트

각 파이프라인 단계별 단위 테스트:

```python
class TestDataCollectionStage:
    def test_markdown_cleaning(self):
        """마크다운 정제 테스트"""
        stage = DataCollectionStage()

        # 테스트 입력
        dirty_markdown = """
        # 제목
        [인쇄](javascript:print())
        내용입니다.
        [**네이버blog**](http://blog.naver.com)
        """

        # 정제
        cleaned = stage.clean_markdown(dirty_markdown)

        # 검증
        assert "[인쇄]" not in cleaned
        assert "[**네이버blog**]" not in cleaned
        assert "내용입니다." in cleaned

    def test_cache_validation(self):
        """캐시 유효성 검증 테스트"""
        stage = DataCollectionStage()

        # 캐시 디렉토리 존재 확인
        assert stage.validate_cache()

        # 파일 존재 확인
        cache_files = list(stage.cache_dir.glob("*.md"))
        assert len(cache_files) > 0
```

### 8.2 통합 테스트

전체 파이프라인 통합 테스트:

```python
class TestRAGPipeline:
    async def test_end_to_end_pipeline(self):
        """종단간 파이프라인 테스트"""
        # 1. 데이터 수집
        data_stage = DataCollectionStage()
        documents = data_stage.load_documents()
        assert len(documents) > 0

        # 2. 청킹
        chunk_stage = ChunkingStage()
        chunks = chunk_stage.create_chunks(documents[:10])
        assert len(chunks) > len(documents)

        # 3. 임베딩
        embed_stage = EmbeddingStage()
        embeddings = await embed_stage.generate_embeddings_batch(
            [chunk.page_content for chunk in chunks]
        )
        assert len(embeddings) == len(chunks)

        # 4. 벡터 스토어
        vector_stage = VectorStoreStage()
        vector_stage.create_index(chunks, embeddings)

        # 5. 검색 테스트
        results = vector_stage.search_with_mmr(
            "서울 지하철 운행 시간",
            k=5
        )
        assert len(results) == 5
```

## 9. 배포 및 실행 가이드

### 9.1 환경 설정

```bash
# Python 3.12 설치 (UV 사용)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12

# 프로젝트 의존성 설치
uv sync

# 환경 변수 설정
export OPENAI_API_KEY="your-api-key"
export OPENROUTER_API_KEY="your-openrouter-key"
```

### 9.2 파이프라인 실행

```bash
# 1. 데이터 크롤링 (이미 완료된 경우 생략)
uv run python src/crawler/seoul_traffic_crawler.py

# 2. RAG 파이프라인 실행
uv run python src/rag_pipeline/run_pipeline.py

# 3. 테스트셋 생성
uv run python src/rag_pipeline/generate_benchmark_50q.py --num-docs 50

# 4. 모델 평가 실행
uv run python src/rag_pipeline/evaluate_models.py \
    --testset data/evaluation/testsets/testset_192aa436bc8960dc.csv \
    --models gpt-4o-mini,exaone-3.5-7.8b,qwen3-8b
```

## 10. 향후 개선 사항

### 10.1 기술적 개선

1. **하이브리드 검색**: BM25 + 벡터 검색 결합
2. **다중 인덱스**: 카테고리별 분리 인덱스
3. **증분 업데이트**: 실시간 문서 추가/삭제
4. **GPU 가속**: FAISS GPU 인덱스 활용

### 10.2 기능 확장

1. **다국어 지원**: 영어, 중국어, 일본어 추가
2. **음성 인터페이스**: STT/TTS 통합
3. **대화형 인터페이스**: 문맥 유지 기능
4. **개인화**: 사용자별 선호도 학습

---

*본 기술 문서는 석사 논문의 보충 자료로서, 구현의 세부 사항과 기술적 결정 사항을 상세히 기록하였다.*