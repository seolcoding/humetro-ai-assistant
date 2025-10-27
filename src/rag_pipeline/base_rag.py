"""
Base RAG Pipeline for Humetro AI Assistant Research
Implements 4-level hierarchical RAG comparison:
1. Baseline (No RAG)
2. Naive RAG (Vector Search)
3. Advanced RAG (Hybrid Search)
4. Graph RAG (Multi-hop Reasoning)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RAGQuery:
    """Query object for RAG pipeline"""
    text: str
    metadata: Optional[Dict[str, Any]] = None
    top_k: int = 5
    temperature: float = 0.7


@dataclass
class RAGResponse:
    """Response object from RAG pipeline"""
    answer: str
    contexts: List[str]
    scores: List[float]
    metadata: Dict[str, Any]
    latency_ms: float


class BaseRAG(ABC):
    """Abstract base class for RAG implementations"""

    def __init__(
        self,
        retriever: Any,
        reranker: Optional[Any] = None,
        generator: Any = None,
        config: Dict[str, Any] = None
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.config = config or {}
        self.name = self.__class__.__name__

    @abstractmethod
    def retrieve(self, query: RAGQuery) -> List[Dict[str, Any]]:
        """Retrieve relevant documents"""
        pass

    @abstractmethod
    def generate(self, query: RAGQuery, contexts: List[str]) -> str:
        """Generate answer based on contexts"""
        pass

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank retrieved documents"""
        if self.reranker:
            return self.reranker.rerank(query, documents)
        return documents

    def __call__(self, query: RAGQuery) -> RAGResponse:
        """Execute RAG pipeline"""
        import time
        start_time = time.time()

        # Retrieve
        retrieved_docs = self.retrieve(query)

        # Rerank (if available)
        if self.reranker:
            retrieved_docs = self.rerank(query.text, retrieved_docs)

        # Extract contexts
        contexts = [doc.get('content', '') for doc in retrieved_docs]
        scores = [doc.get('score', 0.0) for doc in retrieved_docs]

        # Generate
        answer = self.generate(query, contexts)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        return RAGResponse(
            answer=answer,
            contexts=contexts,
            scores=scores,
            metadata={
                'method': self.name,
                'num_contexts': len(contexts),
                'config': self.config
            },
            latency_ms=latency_ms
        )


class BaselineRAG(BaseRAG):
    """Baseline: Pure LLM without retrieval"""

    def retrieve(self, query: RAGQuery) -> List[Dict[str, Any]]:
        """No retrieval for baseline"""
        return []

    def generate(self, query: RAGQuery, contexts: List[str]) -> str:
        """Direct generation without context"""
        prompt = f"질문: {query.text}\n답변:"
        return self.generator.generate(prompt, temperature=query.temperature)


class NaiveRAG(BaseRAG):
    """Naive RAG: Simple vector search"""

    def retrieve(self, query: RAGQuery) -> List[Dict[str, Any]]:
        """Vector similarity search"""
        return self.retriever.search(
            query.text,
            top_k=query.top_k,
            metadata_filter=query.metadata
        )

    def generate(self, query: RAGQuery, contexts: List[str]) -> str:
        """Generate with retrieved contexts"""
        context_text = "\n\n".join(contexts)
        prompt = f"""다음 문맥을 참고하여 질문에 답하세요.

문맥:
{context_text}

질문: {query.text}
답변:"""
        return self.generator.generate(prompt, temperature=query.temperature)


class AdvancedRAG(BaseRAG):
    """Advanced RAG: Hybrid search (BM25 + Semantic)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bm25_weight = self.config.get('bm25_weight', 0.3)
        self.semantic_weight = self.config.get('semantic_weight', 0.7)

    def retrieve(self, query: RAGQuery) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining BM25 and semantic search"""
        # BM25 search
        bm25_results = self.retriever.bm25_search(
            query.text,
            top_k=query.top_k * 2
        )

        # Semantic search
        semantic_results = self.retriever.semantic_search(
            query.text,
            top_k=query.top_k * 2
        )

        # Merge and reweight results
        merged_results = self._merge_results(
            bm25_results,
            semantic_results,
            self.bm25_weight,
            self.semantic_weight
        )

        return merged_results[:query.top_k]

    def _merge_results(
        self,
        bm25_results: List[Dict],
        semantic_results: List[Dict],
        bm25_weight: float,
        semantic_weight: float
    ) -> List[Dict[str, Any]]:
        """Merge BM25 and semantic search results"""
        # Implementation of reciprocal rank fusion
        doc_scores = {}

        for rank, doc in enumerate(bm25_results):
            doc_id = doc['id']
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + bm25_weight / (rank + 1)

        for rank, doc in enumerate(semantic_results):
            doc_id = doc['id']
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + semantic_weight / (rank + 1)

        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Retrieve full documents
        results = []
        for doc_id, score in sorted_docs:
            # Get document from retriever
            doc = self.retriever.get_document(doc_id)
            if doc:
                doc['score'] = score
                results.append(doc)

        return results

    def generate(self, query: RAGQuery, contexts: List[str]) -> str:
        """Generate with hybrid retrieved contexts"""
        context_text = "\n\n".join(contexts)
        prompt = f"""다음 문맥을 참고하여 질문에 답하세요.

문맥:
{context_text}

질문: {query.text}
답변:"""
        return self.generator.generate(prompt, temperature=query.temperature)


class GraphRAG(BaseRAG):
    """Graph RAG: Knowledge graph-based retrieval with multi-hop reasoning"""

    def __init__(self, *args, graph_db=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_db = graph_db
        self.max_hops = self.config.get('max_hops', 2)
        self.relation_weight = self.config.get('relation_weight', 0.5)

    def retrieve(self, query: RAGQuery) -> List[Dict[str, Any]]:
        """Graph-based retrieval with multi-hop reasoning"""
        # 1. Extract entities from query
        entities = self._extract_entities(query.text)

        # 2. Find relevant nodes in graph
        relevant_nodes = []
        for entity in entities:
            nodes = self.graph_db.find_nodes(entity, max_hops=self.max_hops)
            relevant_nodes.extend(nodes)

        # 3. Extract subgraph
        subgraph = self.graph_db.extract_subgraph(relevant_nodes)

        # 4. Rank nodes by relevance
        ranked_nodes = self._rank_nodes(query.text, subgraph)

        # 5. Convert to documents
        documents = []
        for node in ranked_nodes[:query.top_k]:
            documents.append({
                'id': node['id'],
                'content': node['content'],
                'score': node['relevance_score'],
                'metadata': {
                    'node_type': node['type'],
                    'relations': node.get('relations', [])
                }
            })

        return documents

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from query text"""
        # Use NER or keyword extraction
        # For now, simple implementation
        import re
        # Extract Korean nouns and important terms
        entities = re.findall(r'[가-힣]+', text)
        return [e for e in entities if len(e) > 1]

    def _rank_nodes(self, query: str, subgraph: Dict) -> List[Dict]:
        """Rank nodes by relevance to query"""
        # Implement PageRank-style algorithm with query relevance
        # Simplified version for now
        nodes = subgraph.get('nodes', [])

        for node in nodes:
            # Calculate relevance score
            content_similarity = self._calculate_similarity(query, node['content'])
            relation_score = len(node.get('relations', [])) * self.relation_weight
            node['relevance_score'] = content_similarity + relation_score

        return sorted(nodes, key=lambda x: x['relevance_score'], reverse=True)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        # Use embeddings or simple overlap
        # Simplified for now
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def generate(self, query: RAGQuery, contexts: List[str]) -> str:
        """Generate with graph-enhanced contexts"""
        # Include relationship information in prompt
        enhanced_contexts = []
        for i, ctx in enumerate(contexts):
            metadata = self.retriever.get_metadata(i)
            if metadata and 'relations' in metadata:
                relations = metadata['relations']
                enhanced_ctx = f"{ctx}\n[관련 정보: {', '.join(relations)}]"
                enhanced_contexts.append(enhanced_ctx)
            else:
                enhanced_contexts.append(ctx)

        context_text = "\n\n".join(enhanced_contexts)
        prompt = f"""다음 문맥과 관계 정보를 참고하여 질문에 답하세요.

문맥:
{context_text}

질문: {query.text}
답변:"""
        return self.generator.generate(prompt, temperature=query.temperature)