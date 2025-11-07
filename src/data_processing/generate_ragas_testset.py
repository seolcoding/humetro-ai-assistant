#!/usr/bin/env python3
"""
RAGAS 0.3.1+ Korean Testset Generation
Uses latest Knowledge Graph-based approach for generating Korean Q/A pairs
"""

import os
from pathlib import Path
from typing import List, Optional
import pandas as pd
from dotenv import load_dotenv

# RAGAS 0.3.1+ imports
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms import HeadlinesExtractor, HeadlineSplitter, KeyphrasesExtractor
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader


class KoreanRagasTestsetGenerator:
    """
    Latest RAGAS API를 사용한 한국어 Q/A 데이터셋 생성기

    Features:
    - Knowledge Graph 기반 접근
    - 한국어 Persona 지원
    - Transform 기반 노드 처리
    - Single-hop 및 Multi-hop 쿼리 생성
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        """
        Initialize Korean RAGAS testset generator

        Args:
            llm_model: OpenAI model for query generation
            embedding_model: OpenAI embedding model
            api_key: OpenAI API key (loads from env if not provided)
        """
        # Load API key
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")

        # Initialize LLM (no temperature for GPT-5+ models)
        base_llm = ChatOpenAI(
            model=llm_model,
            api_key=api_key
        )
        self.llm = LangchainLLMWrapper(base_llm)

        # Initialize embeddings
        base_embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=api_key
        )
        self.embeddings = LangchainEmbeddingsWrapper(base_embeddings)

        # Korean personas
        self.personas = self._create_korean_personas()

    def _create_korean_personas(self) -> List[Persona]:
        """
        한국어 질문 생성을 위한 Persona 정의

        IMPORTANT: 모든 Persona는 반드시 한국어로 질문하고 답변해야 함을 명시
        """
        return [
            Persona(
                name="지하철 처음 이용자",
                role_description="""
                지하철을 처음 이용하는 승객입니다.
                모든 질문과 답변은 반드시 한국어로 작성되어야 합니다.
                기본적인 이용 방법, 요금, 노선 정보 등에 대해 자세한 안내가 필요합니다.
                질문은 구체적이고 단계별 설명을 요구하는 형태로 작성됩니다.
                예: "지하철 요금은 어떻게 계산되나요?", "환승은 어떻게 하나요?"
                """
            ),
            Persona(
                name="자주 이용하는 승객",
                role_description="""
                지하철을 자주 이용하는 승객입니다.
                모든 질문과 답변은 반드시 한국어로 작성되어야 합니다.
                효율적인 이동, 시간 절약, 편의시설 위치 등 실용적인 정보에 관심이 많습니다.
                질문은 간결하고 핵심적인 정보를 요구하는 형태로 작성됩니다.
                예: "가장 빠른 환승 경로는?", "막차 시간은 언제인가요?"
                """
            ),
            Persona(
                name="불만을 가진 승객",
                role_description="""
                서비스나 시설에 불만을 가진 승객입니다.
                모든 질문과 답변은 반드시 한국어로 작성되어야 합니다.
                문제 상황에 대한 즉각적인 해결책과 보상을 요구합니다.
                질문은 다소 감정적이고 즉각적인 조치를 요구하는 형태로 작성됩니다.
                예: "지연 보상은 어떻게 받나요?", "불편사항을 어디에 신고하나요?"
                """
            ),
            Persona(
                name="교통약자",
                role_description="""
                어르신, 장애인, 임산부 등 교통약자입니다.
                모든 질문과 답변은 반드시 한국어로 작성되어야 합니다.
                접근성, 편의시설, 도움 서비스 등에 대한 정보가 필요합니다.
                질문은 이해하기 쉽고 배려를 요구하는 형태로 작성됩니다.
                예: "휠체어로 이용 가능한가요?", "도움이 필요하면 어디로 연락하나요?"
                """
            )
        ]

    def load_documents(self, docs_dir: str) -> List:
        """
        Load Korean markdown documents from directory

        Args:
            docs_dir: Directory containing markdown files

        Returns:
            List of loaded documents
        """
        print(f"📄 Loading documents from: {docs_dir}")
        loader = DirectoryLoader(
            docs_dir,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} documents")
        return documents

    def create_knowledge_graph(self, documents: List) -> KnowledgeGraph:
        """
        Create Knowledge Graph from documents

        Args:
            documents: List of LangChain documents

        Returns:
            KnowledgeGraph with document nodes
        """
        print(f"🔨 Creating Knowledge Graph...")
        kg = KnowledgeGraph()

        for doc in documents:
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.page_content,
                        "document_metadata": doc.metadata
                    }
                )
            )

        print(f"✅ Created KG with {len(kg.nodes)} nodes")
        return kg

    def apply_transforms(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """
        Apply transforms to enrich Knowledge Graph

        Transforms:
        1. HeadlinesExtractor: Extract section headlines
        2. HeadlineSplitter: Split documents by headlines
        3. KeyphrasesExtractor: Extract key phrases

        Args:
            kg: Knowledge Graph to transform

        Returns:
            Transformed Knowledge Graph
        """
        print(f"🔄 Applying transforms to Knowledge Graph...")

        transforms = [
            HeadlinesExtractor(llm=self.llm, max_num=20),
            HeadlineSplitter(max_tokens=1500),
            KeyphrasesExtractor(llm=self.llm)
        ]

        apply_transforms(kg, transforms=transforms)

        print(f"✅ Transforms applied. Total nodes: {len(kg.nodes)}")
        return kg

    def generate_testset(
        self,
        kg: KnowledgeGraph,
        testset_size: int = 50,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate Korean Q/A testset using Knowledge Graph

        Args:
            kg: Transformed Knowledge Graph
            testset_size: Number of Q/A pairs to generate
            output_path: Optional path to save CSV

        Returns:
            DataFrame with Korean Q/A pairs
        """
        print(f"\n🎯 Generating Korean testset...")
        print(f"   - Target size: {testset_size}")
        print(f"   - Personas: {len(self.personas)}")

        # Define query distribution
        # 50% headline-based, 50% keyphrase-based
        query_distribution = [
            (
                SingleHopSpecificQuerySynthesizer(
                    llm=self.llm,
                    property_name="headlines"
                ),
                0.5
            ),
            (
                SingleHopSpecificQuerySynthesizer(
                    llm=self.llm,
                    property_name="keyphrases"
                ),
                0.5
            ),
        ]

        # Create generator
        generator = TestsetGenerator(
            llm=self.llm,
            embedding_model=self.embeddings,
            knowledge_graph=kg,
            persona_list=self.personas
        )

        # Generate testset
        print(f"⏳ Generating {testset_size} Korean Q/A pairs...")
        testset = generator.generate(
            testset_size=testset_size,
            query_distribution=query_distribution
        )

        # Convert to DataFrame
        testset_df = testset.to_pandas()

        print(f"\n✅ Korean testset generated!")
        print(f"   - Total samples: {len(testset_df)}")
        print(f"   - Columns: {testset_df.columns.tolist()}")

        # Display sample
        if len(testset_df) > 0:
            print(f"\n📝 Sample Korean Q/A:")
            sample = testset_df.iloc[0]
            print(f"   Question: {sample['user_input'][:100]}...")
            print(f"   Answer: {sample['reference'][:100]}...")

        # Save if output path provided
        if output_path:
            testset_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"💾 Saved to: {output_path}")

        return testset_df


def main():
    """
    Main function to generate Korean RAGAS testset
    """
    # Configuration
    ROOT_DIR = Path.cwd()
    DOCS_DIR = ROOT_DIR / "data" / "crawled" / "seoul_traffic" / "markdown_deduplicated"
    OUTPUT_DIR = ROOT_DIR / "data" / "processed"
    OUTPUT_PATH = OUTPUT_DIR / "ragas_korean_testset.csv"

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RAGAS 0.3.1+ Korean Testset Generation")
    print("=" * 60)

    # Initialize generator
    print("\n📦 Initializing Korean RAGAS generator...")
    generator = KoreanRagasTestsetGenerator(
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small"
    )

    # Load documents
    documents = generator.load_documents(str(DOCS_DIR))

    # Create Knowledge Graph
    kg = generator.create_knowledge_graph(documents)

    # Apply transforms
    kg = generator.apply_transforms(kg)

    # Generate testset
    testset_df = generator.generate_testset(
        kg=kg,
        testset_size=50,
        output_path=str(OUTPUT_PATH)
    )

    print("\n" + "=" * 60)
    print("✅ Korean testset generation complete!")
    print("=" * 60)
    print(f"Output: {OUTPUT_PATH}")
    print(f"Total samples: {len(testset_df)}")


if __name__ == "__main__":
    main()
