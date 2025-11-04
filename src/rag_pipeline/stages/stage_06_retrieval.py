"""
Stage 6: Retrieval

RAG retrieval chain using LangChain.
Combines retrieval with LLM generation for question answering.
"""

from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.common.logger import RAGLogger


class RetrievalStage:
    """
    Retrieval Stage for RAG Pipeline.

    Implements RAG chain with LangChain:
    - Retriever for document lookup
    - Prompt template for context formatting
    - LLM for answer generation
    - Chain orchestration

    Features:
    - Configurable retrieval parameters (k, search type)
    - Customizable prompt templates
    - Korean-optimized prompts
    - Source citation support
    """

    # Default Korean RAG prompt
    DEFAULT_PROMPT_TEMPLATE = """당신은 서울시 교통 정보 전문 상담사입니다.
아래 제공된 문서 내용을 바탕으로 질문에 정확하게 답변해주세요.

**중요 지침:**
- 제공된 문서 내용만을 기반으로 답변하세요
- 문서에 없는 내용은 추측하지 마세요
- 답변할 수 없는 경우 솔직하게 "제공된 정보로는 답변하기 어렵습니다"라고 말하세요
- 가능한 경우 구체적인 출처나 관련 부서 정보를 포함하세요

**참고 문서:**
{context}

**질문:** {question}

**답변:**"""

    def __init__(
        self,
        retriever,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        prompt_template: Optional[str] = None,
        logger: Optional[RAGLogger] = None
    ):
        """
        Initialize Retrieval Stage.

        Args:
            retriever: LangChain retriever (from Stage 5)
            model: OpenAI chat model
            temperature: LLM temperature (0.0 = deterministic)
            prompt_template: Custom prompt template (optional)
            logger: Optional logger instance
        """
        self.retriever = retriever
        self.model_name = model
        self.temperature = temperature
        self.logger = logger

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature
        )

        # Create prompt
        prompt_text = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.prompt = ChatPromptTemplate.from_template(prompt_text)

        # Build RAG chain
        self.chain = self._build_chain()

        if self.logger:
            self.logger.info(
                f"RetrievalStage initialized: model={model}, "
                f"temperature={temperature}"
            )

    def _build_chain(self):
        """
        Build LangChain RAG chain.

        Returns:
            Configured RAG chain
        """
        # Format retrieved documents into context string
        def format_docs(docs: List[Document]) -> str:
            """Format documents for prompt context."""
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'unknown')
                content = doc.page_content
                formatted.append(f"[문서 {i}] (출처: {source})\n{content}")
            return "\n\n".join(formatted)

        # RAG chain: retrieve → format → prompt → LLM → parse
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def query(self, question: str) -> str:
        """
        Query RAG system with a question.

        Args:
            question: User question

        Returns:
            Generated answer

        Example:
            answer = stage.query("서울시 대중교통 요금은 얼마인가요?")
        """
        if self.logger:
            self.logger.info(f"Query: {question}")

        answer = self.chain.invoke(question)

        if self.logger:
            self.logger.info(f"Answer generated: {len(answer)} chars")

        return answer

    def query_with_sources(self, question: str) -> Dict[str, Any]:
        """
        Query RAG system and return answer with source documents.

        Args:
            question: User question

        Returns:
            Dictionary with answer and source documents

        Example:
            result = stage.query_with_sources("질문")
            print(result['answer'])
            print(result['sources'])
        """
        if self.logger:
            self.logger.info(f"Query with sources: {question}")

        # Retrieve source documents
        source_docs = self.retriever.invoke(question)

        # Generate answer
        answer = self.chain.invoke(question)

        if self.logger:
            self.logger.info(
                f"Answer generated with {len(source_docs)} source documents"
            )

        return {
            "question": question,
            "answer": answer,
            "sources": source_docs,
            "num_sources": len(source_docs)
        }

    def batch_query(self, questions: List[str]) -> List[str]:
        """
        Process multiple questions in batch.

        Args:
            questions: List of questions

        Returns:
            List of answers (same order as questions)
        """
        if self.logger:
            self.logger.info(f"Batch query: {len(questions)} questions")

        answers = self.chain.batch(questions)

        if self.logger:
            self.logger.info(f"Batch completed: {len(answers)} answers")

        return answers

    def get_chain_info(self) -> dict:
        """
        Get RAG chain configuration info.

        Returns:
            Dictionary with chain metadata
        """
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "has_retriever": self.retriever is not None,
            "prompt_template": self.prompt.template if hasattr(self.prompt, 'template') else None
        }


def create_retrieval_stage(
    retriever,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    prompt_template: Optional[str] = None,
    logger: Optional[RAGLogger] = None
) -> RetrievalStage:
    """
    Factory function to create RetrievalStage.

    Args:
        retriever: LangChain retriever
        model: OpenAI chat model
        temperature: LLM temperature
        prompt_template: Custom prompt template
        logger: Optional logger instance

    Returns:
        Configured RetrievalStage instance
    """
    return RetrievalStage(
        retriever=retriever,
        model=model,
        temperature=temperature,
        prompt_template=prompt_template,
        logger=logger
    )
