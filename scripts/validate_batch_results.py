#!/usr/bin/env python3
"""
Batch Result Validation Script
================================

배치 API 결과 파일(JSONL)을 다각적으로 검증:
1. JSON 구조 검증
2. Schema 준수 확인
3. 문서 품질 분석
4. 통계 및 분포 분석
5. 에러 케이스 탐지
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter


class BatchResultValidator:
    """배치 결과 검증기"""

    def __init__(self, jsonl_path: Path):
        self.jsonl_path = jsonl_path
        self.results = []
        self.stats = defaultdict(int)
        self.errors = []
        self.warnings = []

    def load_results(self):
        """JSONL 파일 로드"""
        print(f"📂 Loading: {self.jsonl_path}")

        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    self.results.append(data)
                except json.JSONDecodeError as e:
                    self.errors.append(f"Line {i}: Invalid JSON - {e}")

        print(f"✅ Loaded {len(self.results)} results")
        print()

    def validate_structure(self):
        """구조 검증"""
        print("=" * 70)
        print("1. STRUCTURE VALIDATION")
        print("=" * 70)

        for i, result in enumerate(self.results, 1):
            # Check top-level keys
            required_keys = ['request', 'status', 'response', 'processed_time']
            missing = [k for k in required_keys if k not in result]

            if missing:
                self.errors.append(f"Result {i}: Missing keys {missing}")
                continue

            # Check response structure
            response = result.get('response', {})
            if not isinstance(response, dict):
                self.errors.append(f"Result {i}: Response is not a dict")
                continue

            if 'candidates' not in response:
                self.errors.append(f"Result {i}: No candidates in response")
                continue

            candidates = response['candidates']
            if not candidates:
                self.errors.append(f"Result {i}: Empty candidates list")
                continue

            # Track finish reasons
            finish_reason = candidates[0].get('finishReason', 'UNKNOWN')
            self.stats[f'finish_reason_{finish_reason}'] += 1

        print(f"✅ Structure validation complete")
        print(f"   Total results: {len(self.results)}")
        print(f"   Errors: {len([e for e in self.errors if 'Result' in e])}")
        print()

    def validate_schema(self):
        """Schema 검증 (7개 required fields)"""
        print("=" * 70)
        print("2. SCHEMA VALIDATION")
        print("=" * 70)

        required_fields = [
            'dialogue_id',
            'original_question',
            'original_answer',
            'topic_path',
            'primary_topic',
            'secondary_topics',
            'document'
        ]

        total_docs = 0
        valid_docs = 0
        field_missing_count = defaultdict(int)

        for i, result in enumerate(self.results, 1):
            try:
                # Extract text from response
                text = result['response']['candidates'][0]['content']['parts'][0]['text']

                # Parse JSON
                try:
                    docs = json.loads(text)
                except json.JSONDecodeError as e:
                    self.errors.append(f"Result {i}: Cannot parse response JSON - {e}")
                    self.stats['json_parse_errors'] += 1
                    continue

                # Check if array
                if not isinstance(docs, list):
                    self.errors.append(f"Result {i}: Response is not an array")
                    self.stats['not_array'] += 1
                    continue

                self.stats['total_dialogues'] += 1
                self.stats['total_documents'] += len(docs)
                total_docs += len(docs)

                # Validate each document
                for doc_idx, doc in enumerate(docs):
                    if not isinstance(doc, dict):
                        self.errors.append(f"Result {i}, Doc {doc_idx}: Not an object")
                        continue

                    # Check required fields
                    missing = [f for f in required_fields if f not in doc]

                    if missing:
                        self.errors.append(f"Result {i}, Doc {doc_idx}: Missing {missing}")
                        for field in missing:
                            field_missing_count[field] += 1
                    else:
                        valid_docs += 1

                        # Check field types
                        if not isinstance(doc.get('secondary_topics'), list):
                            self.warnings.append(f"Result {i}, Doc {doc_idx}: secondary_topics not a list")

                        # Track document lengths
                        doc_len = len(doc.get('document', ''))
                        if 'doc_lengths' not in self.stats:
                            self.stats['doc_lengths'] = []
                        self.stats['doc_lengths'].append(doc_len)

            except Exception as e:
                self.errors.append(f"Result {i}: Unexpected error - {e}")

        print(f"✅ Schema validation complete")
        print(f"   Total documents: {total_docs}")
        print(f"   Valid documents: {valid_docs} ({valid_docs/total_docs*100:.1f}%)")

        if field_missing_count:
            print(f"\n   ⚠️  Missing field counts:")
            for field, count in field_missing_count.items():
                print(f"      - {field}: {count}")
        print()

    def analyze_quality(self):
        """품질 분석"""
        print("=" * 70)
        print("3. QUALITY ANALYSIS")
        print("=" * 70)

        topics_counter = Counter()
        categories_counter = Counter()
        doc_has_frontmatter = 0
        doc_has_markdown_headers = 0

        for i, result in enumerate(self.results, 1):
            try:
                text = result['response']['candidates'][0]['content']['parts'][0]['text']
                docs = json.loads(text)

                for doc in docs:
                    # Topic analysis
                    topics_counter[doc.get('primary_topic', 'N/A')] += 1

                    # Category from topic_path
                    topic_path = doc.get('topic_path', '')
                    if '/' in topic_path:
                        category = topic_path.split('/')[0]
                        categories_counter[category] += 1

                    # Document quality checks
                    document = doc.get('document', '')

                    if document.startswith('---'):
                        doc_has_frontmatter += 1

                    if '##' in document or '# ' in document:
                        doc_has_markdown_headers += 1

            except Exception as e:
                pass

        total_docs = self.stats.get('total_documents', 0)

        print(f"📊 Document Quality:")
        print(f"   - Has YAML front matter: {doc_has_frontmatter}/{total_docs} ({doc_has_frontmatter/total_docs*100:.1f}%)")
        print(f"   - Has Markdown headers: {doc_has_markdown_headers}/{total_docs} ({doc_has_markdown_headers/total_docs*100:.1f}%)")

        if 'doc_lengths' in self.stats and isinstance(self.stats['doc_lengths'], list):
            lengths = self.stats['doc_lengths']
            print(f"\n📏 Document Lengths:")
            print(f"   - Min: {min(lengths)} chars")
            print(f"   - Max: {max(lengths)} chars")
            print(f"   - Avg: {sum(lengths)/len(lengths):.0f} chars")
            print(f"   - Below 500 chars: {len([l for l in lengths if l < 500])}")

        print(f"\n🏷️  Top 10 Primary Topics:")
        for topic, count in topics_counter.most_common(10):
            print(f"   - {topic}: {count}")

        print(f"\n📂 Categories:")
        for category, count in categories_counter.most_common():
            print(f"   - {category}: {count}")

        print()

    def analyze_statistics(self):
        """통계 분석"""
        print("=" * 70)
        print("4. STATISTICS")
        print("=" * 70)

        # Documents per dialogue distribution
        docs_per_dialogue = []

        for result in self.results:
            try:
                text = result['response']['candidates'][0]['content']['parts'][0]['text']
                docs = json.loads(text)
                docs_per_dialogue.append(len(docs))
            except:
                docs_per_dialogue.append(0)

        docs_dist = Counter(docs_per_dialogue)

        print(f"📊 Documents per Dialogue Distribution:")
        for num_docs in sorted(docs_dist.keys()):
            count = docs_dist[num_docs]
            print(f"   - {num_docs} doc(s): {count} dialogues ({count/len(self.results)*100:.1f}%)")

        print(f"\n🔢 Finish Reasons:")
        for key, value in sorted(self.stats.items()):
            if key.startswith('finish_reason_'):
                reason = key.replace('finish_reason_', '')
                print(f"   - {reason}: {value}")

        print()

    def detect_errors(self):
        """에러 탐지"""
        print("=" * 70)
        print("5. ERROR DETECTION")
        print("=" * 70)

        if not self.errors:
            print("✅ No errors detected!")
        else:
            print(f"❌ Found {len(self.errors)} errors:")
            for error in self.errors[:20]:  # Show first 20
                print(f"   - {error}")
            if len(self.errors) > 20:
                print(f"   ... and {len(self.errors) - 20} more")

        print()

        if self.warnings:
            print(f"⚠️  Found {len(self.warnings)} warnings:")
            for warning in self.warnings[:10]:
                print(f"   - {warning}")
            if len(self.warnings) > 10:
                print(f"   ... and {len(self.warnings) - 10} more")
        else:
            print("✅ No warnings")

        print()

    def check_usability(self):
        """활용 가능성 검증"""
        print("=" * 70)
        print("6. USABILITY CHECK")
        print("=" * 70)

        # Can extract QA pairs?
        qa_extractable = 0
        has_complete_qa = 0

        for result in self.results:
            try:
                text = result['response']['candidates'][0]['content']['parts'][0]['text']
                docs = json.loads(text)

                for doc in docs:
                    q = doc.get('original_question', '')
                    a = doc.get('original_answer', '')

                    if q and a:
                        qa_extractable += 1

                        if len(q) > 10 and len(a) > 20:
                            has_complete_qa += 1

            except:
                pass

        total_docs = self.stats.get('total_documents', 0)

        print(f"✅ QA Pair Extraction:")
        print(f"   - Has Q&A: {qa_extractable}/{total_docs} ({qa_extractable/total_docs*100:.1f}%)")
        print(f"   - Complete Q&A: {has_complete_qa}/{total_docs} ({has_complete_qa/total_docs*100:.1f}%)")

        print(f"\n✅ Document Usability:")
        print(f"   - Can be used for RAG: {'YES' if has_complete_qa > total_docs * 0.8 else 'NEEDS REVIEW'}")
        print(f"   - Can build testset: {'YES' if qa_extractable > total_docs * 0.9 else 'NEEDS REVIEW'}")

        print()

    def generate_summary(self):
        """최종 요약"""
        print("=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        total_dialogues = self.stats.get('total_dialogues', 0)
        total_docs = self.stats.get('total_documents', 0)

        print(f"📊 Processing Results:")
        print(f"   - Total dialogues processed: {len(self.results)}")
        print(f"   - Successfully parsed: {total_dialogues}")
        print(f"   - Total documents extracted: {total_docs}")
        print(f"   - Avg docs per dialogue: {total_docs/total_dialogues:.2f}")

        print(f"\n🎯 Quality Metrics:")
        error_rate = len(self.errors) / len(self.results) * 100 if self.results else 0
        print(f"   - Error rate: {error_rate:.1f}%")
        print(f"   - Success rate: {100 - error_rate:.1f}%")

        print(f"\n🚀 Recommendation:")
        if error_rate < 5 and total_docs > total_dialogues * 0.8:
            print(f"   ✅ EXCELLENT - Ready for production use")
        elif error_rate < 10:
            print(f"   ⚠️  GOOD - Minor issues, mostly usable")
        else:
            print(f"   ❌ NEEDS WORK - Significant issues detected")

        print()

    def run_all_validations(self):
        """모든 검증 실행"""
        self.load_results()
        self.validate_structure()
        self.validate_schema()
        self.analyze_quality()
        self.analyze_statistics()
        self.detect_errors()
        self.check_usability()
        self.generate_summary()


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_batch_results.py <batch_result.jsonl>")
        sys.exit(1)

    jsonl_path = Path(sys.argv[1])

    if not jsonl_path.exists():
        print(f"❌ File not found: {jsonl_path}")
        sys.exit(1)

    validator = BatchResultValidator(jsonl_path)
    validator.run_all_validations()


if __name__ == "__main__":
    main()
