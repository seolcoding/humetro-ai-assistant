#!/usr/bin/env python3
"""
RAGAS TestSet Generator with Caching & Documentation
=====================================================

논문 작성을 위한 체계적인 테스트셋 생성 시스템

주요 기능:
1. 로컬 캐싱: 동일 설정 재사용시 API 호출 방지
2. 재현성: 프롬프트, 파라미터, 결과 완전 기록
3. 문서화: 휴먼 리더블한 메타데이터 생성

Author: Claude Code
Date: 2024-11-05
"""

import sys
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class TestsetConfig:
    """테스트셋 생성 설정 (재현성을 위한 완전한 기록)"""

    # 모델 설정
    llm_model: str
    llm_temperature: float
    embedding_model: str

    # 데이터 설정
    num_documents: int
    document_source: str

    # 생성 설정
    testset_size: int
    language: str = "korean"

    # 메타데이터
    created_at: str = None
    ragas_version: str = None

    def __post_init__(self):
        """자동 메타데이터 생성"""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.ragas_version is None:
            import ragas
            self.ragas_version = ragas.__version__

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return asdict(self)

    def get_cache_key(self) -> str:
        """캐시 키 생성 (설정 기반 해시)"""
        # created_at, ragas_version 제외 (재현성에 영향 없음)
        config_str = json.dumps({
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "embedding_model": self.embedding_model,
            "num_documents": self.num_documents,
            "document_source": self.document_source,
            "testset_size": self.testset_size,
            "language": self.language,
        }, sort_keys=True)

        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class CachedTestsetGenerator:
    """캐싱 기능이 있는 RAGAS TestSet Generator"""

    def __init__(self, cache_dir: Path = None):
        """
        Args:
            cache_dir: 캐시 저장 디렉토리 (기본: data/evaluation/testsets)
        """
        if cache_dir is None:
            cache_dir = Path("data/evaluation/testsets")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 캐시 디렉토리: {self.cache_dir}")

    def _load_documents(self, source_dir: str, num_docs: int) -> List[Document]:
        """문서 로드

        Args:
            source_dir: 문서 소스 디렉토리
            num_docs: 로드할 문서 개수

        Returns:
            LangChain Document 리스트
        """
        logger.info(f"📄 문서 로드: {source_dir} ({num_docs}개)")

        docs_path = Path(source_dir)

        if not docs_path.exists():
            raise FileNotFoundError(f"문서 디렉토리 없음: {docs_path}")

        # Markdown 파일 로드
        md_files = list(docs_path.glob("*.md"))[:num_docs]

        if len(md_files) < num_docs:
            logger.warning(f"⚠️  요청 {num_docs}개, 실제 {len(md_files)}개 로드")

        documents = []
        for md_file in md_files:
            content = md_file.read_text(encoding='utf-8')
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(md_file.name),
                    "path": str(md_file)
                }
            )
            documents.append(doc)

        logger.info(f"✅ {len(documents)}개 문서 로드 완료")
        return documents

    def _check_cache(self, config: TestsetConfig) -> Optional[Dict[str, Any]]:
        """캐시 확인

        Args:
            config: 테스트셋 설정

        Returns:
            캐시된 데이터 (없으면 None)
        """
        cache_key = config.get_cache_key()
        cache_file = self.cache_dir / f"testset_{cache_key}.json"

        if cache_file.exists():
            logger.info(f"💾 캐시 발견: {cache_file.name}")

            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)

            logger.info(f"✅ 캐시 로드: {len(cached_data['testset'])}개 질문")
            return cached_data

        logger.info(f"🔍 캐시 없음: 새로 생성 필요")
        return None

    def _update_metadata(self, config: TestsetConfig, testset_df: pd.DataFrame,
                         is_latest: bool = True, is_benchmark: bool = False,
                         description: str = ""):
        """메타데이터 파일 업데이트

        Args:
            config: 테스트셋 설정
            testset_df: 생성된 테스트셋
            is_latest: 최신 버전 여부
            is_benchmark: 벤치마크용 여부
            description: 설명
        """
        metadata_file = self.cache_dir / "metadata.json"

        # 기존 메타데이터 로드
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "latest": None,
                "benchmark": None,
                "versions": []
            }

        cache_key = config.get_cache_key()

        # 기존 버전에서 latest/benchmark 플래그 제거
        if is_latest:
            for v in metadata["versions"]:
                v["is_latest"] = False

        if is_benchmark:
            for v in metadata["versions"]:
                v["is_benchmark"] = False

        # 새 버전 정보
        version_info = {
            "cache_key": cache_key,
            "created_at": config.created_at,
            "num_questions": len(testset_df),
            "language": config.language,
            "llm_model": config.llm_model,
            "num_documents": config.num_documents,
            "description": description,
            "is_latest": is_latest,
            "is_benchmark": is_benchmark,
        }

        # 기존 버전 업데이트 또는 추가
        existing_idx = None
        for idx, v in enumerate(metadata["versions"]):
            if v["cache_key"] == cache_key:
                existing_idx = idx
                break

        if existing_idx is not None:
            # 기존 버전 업데이트 (플래그만)
            metadata["versions"][existing_idx].update({
                "is_latest": is_latest,
                "is_benchmark": is_benchmark,
                "description": description if description else metadata["versions"][existing_idx].get("description", ""),
            })
        else:
            # 새 버전 추가
            metadata["versions"].append(version_info)

        # 최신/벤치마크 포인터 업데이트
        if is_latest:
            metadata["latest"] = cache_key
        if is_benchmark:
            metadata["benchmark"] = cache_key

        # 날짜순 정렬 (최신순)
        metadata["versions"].sort(key=lambda x: x["created_at"], reverse=True)

        # 저장
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"📋 메타데이터 업데이트: {metadata_file}")

    def _save_cache(self, config: TestsetConfig, testset_df: pd.DataFrame,
                    is_latest: bool = True, is_benchmark: bool = False,
                    description: str = "") -> Path:
        """캐시 저장

        Args:
            config: 테스트셋 설정
            testset_df: 생성된 테스트셋 DataFrame

        Returns:
            저장된 캐시 파일 경로
        """
        cache_key = config.get_cache_key()
        cache_file = self.cache_dir / f"testset_{cache_key}.json"

        # 완전한 메타데이터 저장
        cache_data = {
            "config": config.to_dict(),
            "cache_key": cache_key,
            "testset": testset_df.to_dict(orient='records'),
            "stats": {
                "num_questions": len(testset_df),
                "avg_context_length": testset_df['reference_contexts'].apply(
                    lambda x: sum(len(c) for c in x) / len(x) if x else 0
                ).mean() if 'reference_contexts' in testset_df else 0,
            }
        }

        # JSON 저장 (휴먼 리더블)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 캐시 저장: {cache_file}")

        # 추가: CSV 저장 (분석 용이)
        csv_file = cache_file.with_suffix('.csv')
        testset_df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"📊 CSV 저장: {csv_file}")

        # 추가: 문서화 파일 생성
        self._generate_documentation(config, cache_file)

        # 메타데이터 업데이트
        self._update_metadata(config, testset_df, is_latest, is_benchmark, description)

        return cache_file

    def _generate_documentation(self, config: TestsetConfig, cache_file: Path):
        """테스트셋 생성 문서화 (논문용)

        Args:
            config: 테스트셋 설정
            cache_file: 캐시 파일 경로
        """
        doc_file = cache_file.with_suffix('.md')

        doc_content = f"""# RAGAS TestSet Generation Report

## Generation Metadata

- **Generated At**: {config.created_at}
- **RAGAS Version**: {config.ragas_version}
- **Cache Key**: {config.get_cache_key()}

## Model Configuration

### LLM (Language Model)
- **Model**: {config.llm_model}
- **Temperature**: {config.llm_temperature}
- **Purpose**: Question generation, scenario creation, persona generation

### Embedding Model
- **Model**: {config.embedding_model}
- **Purpose**: Document clustering, semantic similarity

## Data Configuration

- **Document Source**: `{config.document_source}`
- **Number of Documents**: {config.num_documents}
- **Target Language**: {config.language}
- **Requested TestSet Size**: {config.testset_size}

## Generation Process

RAGAS TestSet Generator follows this pipeline:

1. **Document Processing**
   - Headlines Extraction
   - Section Splitting
   - Summary Generation

2. **Knowledge Graph Construction**
   - Entity Extraction (NER)
   - Theme Identification
   - Relationship Building
   - Clustering (Cosine Similarity)

3. **Persona Generation**
   - Creates diverse user personas based on document themes
   - Each persona represents different question styles

4. **Scenario Generation**
   - Context-aware scenarios for each persona
   - Realistic user interaction contexts

5. **Question-Answer Synthesis**
   - Multi-hop reasoning questions
   - Factual recall questions
   - Contextual understanding questions

## Output Files

- **JSON**: `{cache_file.name}` (Complete data with metadata)
- **CSV**: `{cache_file.with_suffix('.csv').name}` (Tabular format)
- **Documentation**: `{doc_file.name}` (This file)

## Usage

```python
from src.rag_pipeline.testset_generator import CachedTestsetGenerator

generator = CachedTestsetGenerator()
config, testset_df = generator.generate_or_load(
    llm_model="gpt-4o-mini",
    testset_size=50,
    num_documents=20
)
```

## Reproducibility

This testset can be reproduced by using the same:
- Model configuration (LLM + Embeddings)
- Document source and count
- TestSet size

The cache key ensures identical configurations return cached results.

---
*Generated by RAGAS TestSet Generator with Caching*
"""

        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(doc_content)

        logger.info(f"📝 문서 생성: {doc_file}")

    def _create_korean_personas(self) -> List[Persona]:
        """한국어 전용 Persona 생성"""
        personas = [
            Persona(
                name="서울 시민",
                role_description="서울에 거주하는 한국인으로, 서울 대중교통 정보를 한국어로 질문합니다. 일상적인 통근/통학 경로와 교통 상황에 관심이 많습니다."
            ),
            Persona(
                name="외국인 거주자",
                role_description="서울에 거주하는 외국인으로, 서울 대중교통 정보를 한국어로 질문합니다. 한국어를 배우고 있으며 간단명료한 한국어 답변을 선호합니다."
            ),
            Persona(
                name="교통 관심 시민",
                role_description="서울 교통 정책과 노선 변경에 관심이 많은 한국인입니다. 한국어로 상세한 교통 정보를 질문하고 구체적인 답변을 원합니다."
            ),
        ]
        logger.info(f"  ✅ 한국어 전용 Persona 3개 생성")
        return personas

    def generate_or_load(
        self,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.3,
        embedding_model: str = "text-embedding-3-small",
        document_source: str = "data/crawled/seoul_traffic/markdown_deduplicated",
        num_documents: int = 10,
        testset_size: int = 5,
        language: str = "korean",
        force_regenerate: bool = False,
        use_korean_personas: bool = True,
        is_latest: bool = True,
        is_benchmark: bool = False,
        description: str = "",
    ) -> tuple[TestsetConfig, pd.DataFrame]:
        """테스트셋 생성 또는 캐시 로드

        Args:
            llm_model: LLM 모델명
            llm_temperature: LLM temperature
            embedding_model: 임베딩 모델명
            document_source: 문서 소스 디렉토리
            num_documents: 사용할 문서 개수
            testset_size: 생성할 질문 개수
            language: 타겟 언어
            force_regenerate: 캐시 무시하고 재생성

        Returns:
            (config, testset_df) 튜플
        """
        # 설정 객체 생성
        config = TestsetConfig(
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            embedding_model=embedding_model,
            num_documents=num_documents,
            document_source=document_source,
            testset_size=testset_size,
            language=language,
        )

        logger.info("="*60)
        logger.info("  RAGAS TestSet Generator (Cached)".center(60))
        logger.info("="*60)

        # 캐시 확인
        if not force_regenerate:
            cached = self._check_cache(config)
            if cached:
                testset_df = pd.DataFrame(cached['testset'])
                logger.info(f"✅ 캐시에서 로드: {len(testset_df)}개 질문")
                return config, testset_df

        # 새로 생성
        logger.info("\n[생성 시작]")

        # 1. 문서 로드
        documents = self._load_documents(document_source, num_documents)

        # 2. LLM & Embeddings 설정
        logger.info(f"\n[모델 설정]")
        logger.info(f"  LLM: {llm_model} (temperature={llm_temperature})")
        logger.info(f"  Embeddings: {embedding_model}")

        llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)
        embeddings = OpenAIEmbeddings(model=embedding_model)

        llm_wrapper = LangchainLLMWrapper(llm)
        embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings)

        # 3. Persona 설정 (한국어 강제)
        personas = None
        if use_korean_personas:
            logger.info(f"\n[한국어 Persona 생성]")
            personas = self._create_korean_personas()

        # 4. Generator 생성
        logger.info(f"\n[Generator 초기화]")

        # Persona를 Generator 초기화시 전달
        if personas:
            generator = TestsetGenerator(
                llm=llm_wrapper,
                embedding_model=embeddings_wrapper,
                persona_list=personas,  # 한국어 Persona 전달
            )
        else:
            generator = TestsetGenerator(
                llm=llm_wrapper,
                embedding_model=embeddings_wrapper,
            )

        # 5. TestSet 생성
        logger.info(f"\n[TestSet 생성: {testset_size}개 질문 요청]")

        testset = generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=testset_size,
            raise_exceptions=True,
        )

        testset_df = testset.to_pandas()

        logger.info(f"✅ 생성 완료: {len(testset_df)}개 질문")

        # 5. 캐시 저장 (메타데이터 포함)
        self._save_cache(config, testset_df, is_latest, is_benchmark, description)

        return config, testset_df


def main():
    """CLI 테스트용 메인 함수"""
    generator = CachedTestsetGenerator()

    config, testset_df = generator.generate_or_load(
        llm_model="gpt-4o-mini",
        testset_size=5,
        num_documents=10,
    )

    print("\n" + "="*60)
    print("생성된 질문:")
    print("-"*60)

    for idx, row in testset_df.iterrows():
        print(f"\n[질문 {idx+1}]")
        print(f"Q: {row.get('user_input', 'N/A')}")
        print(f"A: {row.get('reference', 'N/A')[:100]}...")

    print("\n" + "="*60)
    print(f"💾 캐시 키: {config.get_cache_key()}")
    print("="*60)


if __name__ == "__main__":
    main()
