#!/usr/bin/env python3
"""
Knowledge Graph Construction for Unstructured Data
Based on agentic_kg_build/8_kg_construction_2.ipynb

Uses Neo4j GraphRAG's SimpleKGPipeline to:
1. Chunk markdown files
2. Extract entities and relationships
3. Create vector embeddings
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.kg_agent.config.llm_config import get_llm, DEFAULT_KG_MODEL
from src.kg_agent.utils.tools import get_data_import_dir
from src.kg_agent.utils.neo4j_for_adk import graphdb

import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Components for Neo4j GraphRAG
# ============================================================================

from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks


class RegexTextSplitter(TextSplitter):
    """Split text using regex matched delimiters."""

    def __init__(self, pattern: str):
        self.pattern = pattern

    async def run(self, text: str) -> TextChunks:
        """Splits a piece of text into chunks.

        Args:
            text (str): The text to be split.

        Returns:
            TextChunks: A list of chunks.
        """
        texts = re.split(self.pattern, text)
        chunks = [TextChunk(text=str(text), index=i) for (i, text) in enumerate(texts)]
        return TextChunks(chunks=chunks)


from neo4j_graphrag.experimental.components.pdf_loader import DataLoader
from neo4j_graphrag.experimental.components.types import PdfDocument, DocumentInfo


class MarkdownDataLoader(DataLoader):
    """Custom loader for markdown files."""

    def extract_title(self, markdown_text):
        """Extract first H1 header as document title"""
        pattern = r'^# (.+)$'
        match = re.search(pattern, markdown_text, re.MULTILINE)
        return match.group(1) if match else "Untitled"

    async def run(self, filepath: Path, metadata={}) -> PdfDocument:
        with open(filepath, "r", encoding='utf-8') as f:
            markdown_text = f.read()
        doc_headline = self.extract_title(markdown_text)
        markdown_info = DocumentInfo(
            path=str(filepath),
            metadata={
                "title": doc_headline,
            }
        )
        return PdfDocument(text=markdown_text, document_info=markdown_info)


# ============================================================================
# Helper Functions
# ============================================================================

def file_context(file_path: str, num_lines: int = 5) -> str:
    """Helper function to extract the first few lines of a file

    Args:
        file_path (str): Path to the file
        num_lines (int, optional): Number of lines to extract. Defaults to 5.

    Returns:
        str: First few lines of the file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = []
        for _ in range(num_lines):
            line = f.readline()
            if not line:
                break
            lines.append(line)
    return "\n".join(lines)


def contextualize_er_extraction_prompt(context: str) -> str:
    """Creates a prompt with pre-amble file content for context during entity+relationship extraction.

    The context is concatenated into the string, which later will be used as a template
    for values like {schema} and {text}.
    """
    general_instructions = """
    You are a top-tier algorithm designed for extracting
    information in structured formats to build a knowledge graph.

    Extract the entities (nodes) and specify their type from the following text.
    Also extract the relationships between these nodes.

    Return result as JSON using the following format:
    {{"nodes": [ {{"id": "0", "label": "Person", "properties": {{"name": "John"}} }}],
    "relationships": [{{"type": "KNOWS", "start_node_id": "0", "end_node_id": "1", "properties": {{"since": "2024-08-01"}} }}] }}

    Use only the following node and relationship types (if provided):
    {schema}

    Assign a unique ID (string) to each node, and reuse it to define relationships.
    Do respect the source and target node types for relationship and
    the relationship direction.

    Make sure you adhere to the following rules to produce valid JSON objects:
    - Do not return any additional information other than the JSON in it.
    - Omit any backticks around the JSON - simply output the JSON on its own.
    - The JSON object must not wrapped into a list - it is its own JSON object.
    - Property names must be enclosed in double quotes
    """

    context_goes_here = f"""
    Consider the following context to help identify entities and relationships:
    <context>
    {context}
    </context>"""

    input_goes_here = """
    Input text:

    {text}
    """

    return general_instructions + "\n" + context_goes_here + "\n" + input_goes_here


# ============================================================================
# KG Builder Factory
# ============================================================================

def make_kg_builder(
    file_path: str,
    approved_entities: List[str],
    approved_fact_types: Dict[str, Any],
    model_name: str = DEFAULT_KG_MODEL
):
    """
    Creates a KG builder pipeline for a given file.

    Args:
        file_path: Path to the markdown file to process
        approved_entities: List of approved entity types from NER agent
        approved_fact_types: Dict of approved fact types from Fact Extraction agent
        model_name: LLM model to use

    Returns:
        SimpleKGPipeline configured for the file
    """
    from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
    from neo4j_graphrag.llm import OpenAILLM
    from neo4j_graphrag.embeddings import OpenAIEmbeddings

    # Setup LLM and embeddings
    llm_for_neo4j = OpenAILLM(model_name=model_name, model_params={"temperature": 0})
    embedder = OpenAIEmbeddings(model="text-embedding-3-large")

    # Get Neo4j driver
    neo4j_driver = graphdb.driver

    # Build entity schema from approved entities and facts
    schema_node_types = approved_entities
    schema_relationship_types = [key.upper() for key in approved_fact_types.keys()]
    schema_patterns = [
        [fact['subject_label'], fact['predicate_label'].upper(), fact['object_label']]
        for fact in approved_fact_types.values()
    ]

    entity_schema = {
        "node_types": schema_node_types,
        "relationship_types": schema_relationship_types,
        "patterns": schema_patterns,
        "additional_node_types": False,  # Strict mode
    }

    # Create contextualized prompt
    context = file_context(file_path)
    contextualized_prompt = contextualize_er_extraction_prompt(context)

    # Build pipeline
    return SimpleKGPipeline(
        llm=llm_for_neo4j,
        driver=neo4j_driver,
        embedder=embedder,
        from_pdf=True,  # Using custom loader
        pdf_loader=MarkdownDataLoader(),
        text_splitter=RegexTextSplitter("---"),  # Split on "---"
        schema=entity_schema,
        prompt_template=contextualized_prompt,
    )


async def construct_kg_from_files(
    approved_files: List[str],
    approved_entities: List[str],
    approved_fact_types: Dict[str, Any],
    model_name: str = DEFAULT_KG_MODEL
) -> Dict[str, Any]:
    """
    Process multiple markdown files and construct knowledge graph.

    Args:
        approved_files: List of file paths relative to import directory
        approved_entities: List of approved entity types
        approved_fact_types: Dict of approved fact types
        model_name: LLM model to use

    Returns:
        Dict with processing results
    """
    import_dir = get_data_import_dir()
    results = []

    for file_name in approved_files:
        file_path = import_dir / file_name

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue

        logger.info(f"Processing file: {file_name}")

        try:
            kg_builder = make_kg_builder(
                str(file_path),
                approved_entities,
                approved_fact_types,
                model_name
            )

            result = await kg_builder.run_async(file_path=str(file_path))
            results.append({
                "file": file_name,
                "status": "success",
                "result": result.result
            })
            logger.info(f"  ✅ Completed: {file_name}")

        except Exception as e:
            logger.error(f"  ❌ Failed: {file_name} - {e}")
            results.append({
                "file": file_name,
                "status": "error",
                "error": str(e)
            })

    logger.info("All files processed.")
    return {
        "total_files": len(approved_files),
        "processed": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "error"]),
        "results": results
    }


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    async def test_kg_construction():
        """Test KG construction with Seoul traffic data"""

        print("=" * 60)
        print("KG Construction Test")
        print("=" * 60)

        # Get first 3 markdown files for testing
        import_dir = get_data_import_dir()
        md_files = sorted([str(f.relative_to(import_dir)) for f in import_dir.rglob("*.md")])[:3]

        print(f"\n1. Testing with {len(md_files)} files:")
        for f in md_files:
            print(f"   - {f}")

        # Use entities and facts from previous agents
        approved_entities = [
            'Station', 'Line', 'Route',
            'TransportationMode', 'Service', 'Event',
            'Landmark', 'Facility', 'SafetyMeasure', 'Schedule'
        ]

        approved_fact_types = {
            'operates_on': {
                'subject_label': 'TransportationMode',
                'predicate_label': 'operates_on',
                'object_label': 'Line'
            },
            'is_part_of': {
                'subject_label': 'Station',
                'predicate_label': 'is_part_of',
                'object_label': 'Line'
            },
            'connects': {
                'subject_label': 'Route',
                'predicate_label': 'connects',
                'object_label': 'Landmark'
            }
        }

        print(f"\n2. Entity types: {len(approved_entities)}")
        print(f"3. Relationship types: {len(approved_fact_types)}")

        # Construct KG
        print("\n4. Starting KG construction...")
        results = await construct_kg_from_files(
            md_files,
            approved_entities,
            approved_fact_types,
            model_name="gpt-4o"
        )

        print("\n" + "=" * 60)
        print("Construction Results:")
        print("=" * 60)
        print(f"Total files: {results['total_files']}")
        print(f"Processed: {results['processed']}")
        print(f"Failed: {results['failed']}")
        print("=" * 60)

        if results['processed'] > 0:
            print("\n✅ Test PASSED: KG construction successful!")
            return True
        else:
            print("\n❌ Test FAILED: No files processed successfully")
            return False

    # Run the test
    asyncio.run(test_kg_construction())
