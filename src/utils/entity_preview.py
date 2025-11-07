"""Simple entity extractor for preview (rule-based)"""

import re
from collections import Counter
from typing import List, Dict, Set


class SimpleEntityExtractor:
    """
    Simple rule-based entity extraction for quick preview.

    This is NOT production-grade NER - just a fast preview for the crawler.
    For actual Knowledge Graph generation, use proper NER (spaCy, Transformers, or LLM).
    """

    def __init__(self):
        # Transport keywords
        self.transport_keywords: Set[str] = {
            '지하철', '버스', '전철', '열차', '노선',
            '1호선', '2호선', '3호선', '4호선', '5호선',
            '6호선', '7호선', '8호선', '9호선',
            '공항철도', 'GTX', 'SRT', 'KTX'
        }

        # Location keywords
        self.location_keywords: Set[str] = {
            '강남', '강북', '서초', '송파', '강동',
            '역', '구', '동', '로', '대로', '서울'
        }

        # Incident keywords
        self.incident_keywords: Set[str] = {
            '사고', '지연', '운행중단', '장애', '공사',
            '통제', '우회', '폐쇄', '차단', '정체'
        }

    def extract_preview(self, text: str, top_n: int = 20) -> List[str]:
        """
        Extract top N keywords as a simple preview.

        Returns most frequent nouns (2-10 characters).
        """
        if not text:
            return []

        # Extract Korean nouns (simple pattern)
        pattern = r'[가-힣]{2,10}'
        matches = re.findall(pattern, text)

        # Count frequencies
        counter = Counter(matches)

        # Return top N
        return [word for word, _ in counter.most_common(top_n)]

    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Simple rule-based named entity extraction.

        Categories:
        - TRANSPORT: Transportation-related entities
        - LOCATION: Location names
        - INCIDENT: Incident/event types
        """
        entities: Dict[str, List[str]] = {
            'TRANSPORT': [],
            'LOCATION': [],
            'INCIDENT': []
        }

        if not text:
            return entities

        # Extract all Korean words
        pattern = r'[가-힣]{2,10}'
        words = re.findall(pattern, text)

        # Classify by keyword matching
        for word in set(words):  # Use set to avoid duplicates
            if any(kw in word for kw in self.transport_keywords):
                entities['TRANSPORT'].append(word)
            elif any(kw in word for kw in self.location_keywords):
                entities['LOCATION'].append(word)
            elif any(kw in word for kw in self.incident_keywords):
                entities['INCIDENT'].append(word)

        return entities


if __name__ == "__main__":
    # Example usage
    extractor = SimpleEntityExtractor()

    sample_text = """
    지하철 2호선 강남역에서 신호 장애로 인한 운행 지연이 발생했습니다.
    서울교통공사는 현재 복구 작업을 진행 중이며, 선릉역까지 우회 운행합니다.
    """

    preview = extractor.extract_preview(sample_text)
    print("Preview:", preview)

    entities = extractor.extract_named_entities(sample_text)
    print("Named Entities:", entities)
