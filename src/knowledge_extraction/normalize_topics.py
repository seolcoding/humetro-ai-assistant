#!/usr/bin/env python3
"""
Topic Normalization and Entity Resolution for Dasan Knowledge Base.

This script consolidates granular topics into a normalized hierarchy using:
1. Text normalization (whitespace, punctuation)
2. Synonym mapping (manual + embedding-based clustering)
3. Hierarchical consolidation (merge similar topics)

Usage:
    uv run python src/knowledge_extraction/normalize_topics.py \
        --input data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl \
        --output data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_normalized.jsonl \
        --analyze-only  # Optional: Only show analysis without applying changes
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter, defaultdict
import argparse
from difflib import SequenceMatcher
import Levenshtein  # For better string similarity


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TopicNormalizer:
    """Normalize and consolidate granular topics using entity resolution."""

    # Level 1: Category consolidation rules
    CATEGORY_MAPPINGS = {
        # Normalize variations to standard categories
        '생활하수도 관련 문의': '생활하수도_관련_문의',
        '상수도_관련_문의': '생활하수도_관련_문의',
        '상수도': '생활하수도_관련_문의',
        '상수도_안내': '생활하수도_관련_문의',
        '수도요금': '생활하수도_관련_문의',
        '수도': '생활하수도_관련_문의',
        '상하수도': '생활하수도_관련_문의',
        '상하수도_요금': '생활하수도_관련_문의',
        '상하수도_관련_문의': '생활하수도_관련_문의',
        '상하수도_민원': '생활하수도_관련_문의',
        '상수도_서비스': '생활하수도_관련_문의',
        '상수도_이용_안내': '생활하수도_관련_문의',
        '수도_관련_문의': '생활하수도_관련_문의',
        '수도_및_배관': '생활하수도_관련_문의',
        '생활상수도_관련_문의': '생활하수도_관련_문의',

        '대중교to_안내': '대중교통_안내',  # typo
        '교통': '대중교통_안내',
        '교통_안내': '대중교통_안내',
        '교통정보': '대중교통_안내',
        '교통정보_안내': '대중교통_안내',
        '기타_교통_안내': '대중교통_안내',
        '도로_교통': '대중교통_안내',
        '도로교통_안내': '대중교통_안내',
        '항공_교통_안내': '대중교통_안내',
        '교통_차량_관련_문의': '대중교통_안내',

        '일반행정': '일반행정_문의',
        '민원신청_안내': '일반행정_문의',
        '공공서비스': '일반행정_문의',
        '공공서비스_안내': '일반행정_문의',
        '생활민원': '일반행정_문의',
        '생활정보': '일반행정_문의',
        '생활정보_안내': '일반행정_문의',
        '생활편의': '일반행정_문의',
        '생활편의_안내': '일반행정_문의',
        '생활': '일반행정_문의',
        '생활환경': '일반행정_문의',
        '복지': '일반행정_문의',
        '복지_서비스': '일반행정_문의',
        '보건_복지': '일반행정_문의',
        '보건_의료': '일반행정_문의',
        '건강_의료': '일반행정_문의',
        '문화': '일반행정_문의',
        '문화_여가': '일반행정_문의',
        '문화_예술': '일반행정_문의',
        '관광정보': '일반행정_문의',
        '주택': '일반행정_문의',
        '주택_및_부동산': '일반행정_문의',
        '주택_건축': '일반행정_문의',
        '일자리_지원': '일반행정_문의',
        '소상공인_지원': '일반행정_문의',
        '환경_위생': '일반행정_문의',
        '긴급신고': '일반행정_문의',
        '긴급신고_및_민원': '일반행정_문의',
        '교통_위반_신고': '일반행정_문의',
        '위치_안내': '일반행정_문의',
        '차량관리': '일반행정_문의',
        '공과금_납부': '일반행정_문의',
    }

    # Level 2: Domain consolidation rules
    DOMAIN_MAPPINGS = {
        # Water/Sewage domains
        '상수도': '수도요금',
        '요금': '수도요금',

        # Transportation domains
        '경로_검색': '교통경로',

        # COVID domains
        '방역지침': '방역수칙',
        '방역_지침': '방역수칙',

        # Welfare domains
        '문화행사': '문화_행사',
        '문화시설': '문화_행사',
    }

    # Level 3: Primary topic consolidation patterns
    TOPIC_PATTERNS = {
        # Water bill topics
        r'수도요금[_\s]*납부.*': '수도요금_납부',
        r'수도요금[_\s]*조회.*': '수도요금_조회',
        r'수도요금[_\s]*온라인.*': '수도요금_온라인_조회',
        r'수도요금[_\s]*명의.*': '수도요금_명의변경',
        r'수도요금[_\s]*이사.*': '수도요금_이사정산',
        r'수도요금[_\s]*감면.*': '수도요금_감면',
        r'수도요금[_\s]*자동.*': '수도요금_자동납부',

        # Bus topics
        r'버스[_\s]*노선.*': '버스_노선안내',
        r'버스[_\s]*운행.*': '버스_운행안내',
        r'버스[_\s]*요금.*': '버스_요금안내',
        r'지역간[_\s]*버스.*': '버스_노선안내',

        # Subway topics
        r'지하철[_\s]*경로.*': '지하철_경로안내',
        r'지하철[_\s]*노선.*': '지하철_노선안내',

        # Route search
        r'경로[_\s]*검색': '교통경로_검색',
        r'지역간[_\s]*경로.*': '교통경로_검색',

        # Disaster support
        r'재난지원금[_\s]*안내': '재난지원금',
        r'재난지원금[_\s]*지급.*': '재난지원금',
    }

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize normalizer.

        Args:
            similarity_threshold: Minimum similarity for auto-clustering (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.stats = {
            'original_categories': 0,
            'normalized_categories': 0,
            'original_domains': 0,
            'normalized_domains': 0,
            'original_topics': 0,
            'normalized_topics': 0,
            'documents_processed': 0,
            'category_consolidations': defaultdict(list),
            'domain_consolidations': defaultdict(list),
            'topic_consolidations': defaultdict(list),
        }

    def normalize_text(self, text: str) -> str:
        """
        Basic text normalization.

        Args:
            text: Raw text

        Returns:
            Normalized text
        """
        if not text:
            return ''

        # Remove extra whitespace
        text = re.sub(r'\s+', '_', text.strip())

        # Remove duplicate underscores
        text = re.sub(r'_+', '_', text)

        return text

    def normalize_category(self, category: str) -> str:
        """
        Normalize category using mapping rules.

        Args:
            category: Original category

        Returns:
            Normalized category
        """
        normalized = self.normalize_text(category)

        # Apply mapping
        if normalized in self.CATEGORY_MAPPINGS:
            mapped = self.CATEGORY_MAPPINGS[normalized]
            self.stats['category_consolidations'][mapped].append(normalized)
            return mapped

        return normalized

    def normalize_domain(self, domain: str) -> str:
        """
        Normalize domain using mapping rules.

        Args:
            domain: Original domain

        Returns:
            Normalized domain
        """
        normalized = self.normalize_text(domain)

        # Apply mapping
        if normalized in self.DOMAIN_MAPPINGS:
            mapped = self.DOMAIN_MAPPINGS[normalized]
            self.stats['domain_consolidations'][mapped].append(normalized)
            return mapped

        return normalized

    def normalize_topic(self, topic: str) -> str:
        """
        Normalize topic using pattern rules.

        Args:
            topic: Original topic

        Returns:
            Normalized topic
        """
        normalized = self.normalize_text(topic)

        # Try pattern matching
        for pattern, replacement in self.TOPIC_PATTERNS.items():
            if re.match(pattern, normalized):
                self.stats['topic_consolidations'][replacement].append(normalized)
                return replacement

        return normalized

    def cluster_similar_topics(
        self,
        topics: Counter,
        min_occurrences: int = 3
    ) -> Dict[str, str]:
        """
        Auto-cluster similar topics using Levenshtein distance.

        Args:
            topics: Counter of topic frequencies
            min_occurrences: Minimum occurrences for canonical form

        Returns:
            Mapping of topic -> canonical_form
        """
        logger.info(f"\nAuto-clustering similar topics (threshold={self.similarity_threshold})...")

        # Filter by frequency: common topics become canonical
        common_topics = [topic for topic, count in topics.items() if count >= min_occurrences]
        rare_topics = [topic for topic, count in topics.items() if count < min_occurrences]

        logger.info(f"  Common topics (>={min_occurrences} docs): {len(common_topics)}")
        logger.info(f"  Rare topics (<{min_occurrences} docs): {len(rare_topics)}")

        # Build clustering map
        clustering_map = {}
        total_merged = 0

        for rare in rare_topics:
            best_match = None
            best_similarity = 0.0

            # Find most similar common topic
            for common in common_topics:
                similarity = Levenshtein.ratio(rare, common)
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = common

            if best_match:
                clustering_map[rare] = best_match
                self.stats['topic_consolidations'][best_match].append(rare)
                total_merged += 1

        logger.info(f"  Auto-merged {total_merged} rare topics into common topics")

        return clustering_map

    def apply_topic_clustering(
        self,
        input_path: Path,
        output_path: Path
    ):
        """
        Second pass: Apply auto-clustering to rare topics.

        Args:
            input_path: Input JSONL (from first pass)
            output_path: Output JSONL (with clustering applied)
        """
        logger.info("\n" + "=" * 70)
        logger.info("TOPIC CLUSTERING (Second Pass)")
        logger.info("=" * 70)

        # Collect topic frequencies from first pass
        topics = Counter()
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                topics[doc.get('primary_topic', '')] += 1

        # Auto-cluster
        clustering_map = self.cluster_similar_topics(topics, min_occurrences=3)

        if not clustering_map:
            logger.info("No additional clustering needed")
            return

        # Apply clustering
        logger.info(f"\nApplying clustering to {input_path}...")
        clustered_docs = []

        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                doc = json.loads(line)

                # Update primary_topic if clustered
                topic = doc.get('primary_topic', '')
                if topic in clustering_map:
                    doc['primary_topic'] = clustering_map[topic]

                    # Update topic_path level 3
                    path_parts = doc.get('topic_path', '').split('/')
                    if len(path_parts) >= 3:
                        path_parts[2] = clustering_map[topic]
                        doc['topic_path'] = '/'.join(path_parts)

                clustered_docs.append(doc)

        # Write output
        logger.info(f"Writing clustered documents to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in clustered_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

        logger.info(f"✓ Clustering complete")

    def analyze_topics(self, jsonl_path: Path) -> Dict:
        """
        Analyze topic distribution before normalization.

        Args:
            jsonl_path: Input JSONL file

        Returns:
            Analysis statistics
        """
        categories = Counter()
        domains = Counter()
        topics = Counter()

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)

                # Extract hierarchy
                path = doc.get('topic_path', '')
                parts = path.split('/')

                if len(parts) >= 1:
                    categories[parts[0]] += 1
                if len(parts) >= 2:
                    domains[parts[1]] += 1

                topics[doc.get('primary_topic', '')] += 1

        self.stats['original_categories'] = len(categories)
        self.stats['original_domains'] = len(domains)
        self.stats['original_topics'] = len(topics)

        return {
            'categories': categories,
            'domains': domains,
            'topics': topics
        }

    def normalize_document(self, doc: Dict) -> Dict:
        """
        Normalize topic hierarchy in document.

        Args:
            doc: Original document

        Returns:
            Normalized document
        """
        # Extract current hierarchy
        topic_path = doc.get('topic_path', '')
        parts = topic_path.split('/')

        # Normalize each level
        normalized_parts = []
        if len(parts) >= 1:
            normalized_parts.append(self.normalize_category(parts[0]))
        if len(parts) >= 2:
            normalized_parts.append(self.normalize_domain(parts[1]))
        if len(parts) >= 3:
            normalized_parts.append(self.normalize_topic(parts[2]))

        # Update document
        doc['topic_path'] = '/'.join(normalized_parts)
        doc['primary_topic'] = self.normalize_topic(doc.get('primary_topic', ''))

        # Update metadata category/domain
        if 'metadata' in doc:
            if len(normalized_parts) >= 1:
                doc['metadata']['category'] = normalized_parts[0]
            if len(normalized_parts) >= 2:
                doc['metadata']['domain'] = normalized_parts[1]

        return doc

    def process(
        self,
        input_path: Path,
        output_path: Path,
        analyze_only: bool = False
    ) -> Dict:
        """
        Process JSONL with topic normalization.

        Args:
            input_path: Input JSONL
            output_path: Output JSONL
            analyze_only: Only analyze, don't write

        Returns:
            Processing statistics
        """
        logger.info("=" * 70)
        logger.info("TOPIC NORMALIZATION")
        logger.info("=" * 70)

        # Analyze before
        logger.info("Analyzing original topic distribution...")
        before = self.analyze_topics(input_path)

        logger.info(f"Original hierarchy:")
        logger.info(f"  Categories: {len(before['categories'])}")
        logger.info(f"  Domains: {len(before['domains'])}")
        logger.info(f"  Primary topics: {len(before['topics'])}")

        if analyze_only:
            logger.info("\nAnalysis-only mode - skipping normalization")
            return self.stats

        # Process documents
        logger.info(f"\nNormalizing documents from {input_path}...")

        normalized_docs = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 1000 == 0:
                    logger.info(f"Processed {line_num:,} documents...")

                doc = json.loads(line)
                normalized_doc = self.normalize_document(doc)
                normalized_docs.append(normalized_doc)
                self.stats['documents_processed'] += 1

        # Write output
        logger.info(f"\nWriting normalized documents to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in normalized_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

        # Analyze after
        logger.info("Analyzing normalized topic distribution...")
        after = self.analyze_topics(output_path)

        self.stats['normalized_categories'] = len(after['categories'])
        self.stats['normalized_domains'] = len(after['domains'])
        self.stats['normalized_topics'] = len(after['topics'])

        # Log summary
        self._log_summary()

        return self.stats

    def _log_summary(self):
        """Log normalization summary."""
        logger.info("=" * 70)
        logger.info("NORMALIZATION SUMMARY")
        logger.info("=" * 70)

        logger.info(f"Documents processed: {self.stats['documents_processed']:,}")
        logger.info("")

        logger.info("Consolidation results:")
        logger.info(f"  Categories: {self.stats['original_categories']} → {self.stats['normalized_categories']} "
                   f"({self.stats['original_categories'] - self.stats['normalized_categories']} merged)")
        logger.info(f"  Domains: {self.stats['original_domains']} → {self.stats['normalized_domains']} "
                   f"({self.stats['original_domains'] - self.stats['normalized_domains']} merged)")
        logger.info(f"  Topics: {self.stats['original_topics']} → {self.stats['normalized_topics']} "
                   f"({self.stats['original_topics'] - self.stats['normalized_topics']} merged)")
        logger.info("")

        logger.info("Category consolidations:")
        for target, sources in sorted(self.stats['category_consolidations'].items()):
            if sources:
                logger.info(f"  {target}:")
                for src in list(set(sources))[:5]:  # Show first 5
                    logger.info(f"    ← {src}")

        logger.info("")
        logger.info("Domain consolidations:")
        for target, sources in sorted(self.stats['domain_consolidations'].items()):
            if sources:
                logger.info(f"  {target}:")
                for src in list(set(sources))[:5]:
                    logger.info(f"    ← {src}")

        logger.info("")
        logger.info("Top 10 topic consolidations:")
        for target, sources in sorted(
            self.stats['topic_consolidations'].items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:10]:
            logger.info(f"  {target}: {len(set(sources))} variants merged")

        logger.info("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Normalize and consolidate granular topics'
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input JSONL file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_normalized.jsonl'),
        help='Output normalized JSONL file'
    )
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze distribution without normalizing'
    )
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.85,
        help='Similarity threshold for auto-clustering (0-1)'
    )
    parser.add_argument(
        '--no-clustering',
        action='store_true',
        help='Skip second pass (auto-clustering)'
    )

    args = parser.parse_args()

    # Run normalization (Pass 1: Rule-based)
    normalizer = TopicNormalizer(similarity_threshold=args.similarity_threshold)
    stats = normalizer.process(args.input, args.output, args.analyze_only)

    if args.analyze_only:
        return 0

    # Run clustering (Pass 2: Similarity-based)
    if not args.no_clustering:
        temp_output = args.output.with_suffix('.temp.jsonl')
        args.output.rename(temp_output)  # Pass 1 result → temp
        normalizer.apply_topic_clustering(temp_output, args.output)
        temp_output.unlink()  # Remove temp file

        # Final analysis
        logger.info("\n" + "=" * 70)
        logger.info("FINAL ANALYSIS")
        logger.info("=" * 70)
        final_stats = normalizer.analyze_topics(args.output)
        logger.info(f"Final hierarchy:")
        logger.info(f"  Categories: {len(final_stats['categories'])}")
        logger.info(f"  Domains: {len(final_stats['domains'])}")
        logger.info(f"  Primary topics: {len(final_stats['topics'])}")

        # Calculate reduction
        original_topics = stats['original_topics']
        final_topics = len(final_stats['topics'])
        reduction = (1 - final_topics / original_topics) * 100

        logger.info("")
        logger.info(f"✓ Topic reduction: {original_topics:,} → {final_topics:,} ({reduction:.1f}% reduction)")
    else:
        logger.info(f"\n✓ Normalization complete: {args.output}")
        logger.info(f"✓ Reduced {stats['original_topics']:,} topics to {stats['normalized_topics']:,} "
                   f"({(1 - stats['normalized_topics']/stats['original_topics'])*100:.1f}% reduction)")

    return 0


if __name__ == '__main__':
    exit(main())
