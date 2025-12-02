"""
E2E 테스트: GPT-5.1 질문 생성 → AutoRAG 파이프라인 통합 검증

1. GPT-5.1로 10개 질문 생성 (실제 API 호출)
2. AutoRAG 포맷으로 변환 (qa.parquet, corpus.parquet)
3. GPT-4o-mini로 답변 생성 (기존 파이프라인)
4. RAGAS 평가 (Faithfulness, Answer Relevancy)

사용법:
    python run_e2e_test.py
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import pandas as pd

# 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from doc_selectors.multihop_selector import MultihopDocumentSelector, DocumentPair, DocumentTriple
from converters.autorag_converter import AutoRAGConverter, AutoRAGQA
from prompts.system_prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_MULTIHOP
from prompts.question_prompts import PROMPTS

# OpenAI 임포트
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    print("Warning: openai 패키지가 없습니다. pip install openai")
    HAS_OPENAI = False


class GPT51QuestionGenerator:
    """GPT-5.1 질문 생성기 (실제 API 호출)"""

    def __init__(
        self,
        model: str = "gpt-5.1",
        reasoning_effort: str = "medium",
        verbosity: str = "medium",
        fallback_model: str = "gpt-4o-mini"  # GPT-5.1 없으면 폴백
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.fallback_model = fallback_model

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다")

        self.client = OpenAI(api_key=api_key)
        self.stats = {"calls": 0, "success": 0, "failed": 0, "tokens": 0}

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        retries: int = 2
    ) -> Optional[str]:
        """API 호출"""
        self.stats["calls"] += 1

        for attempt in range(retries + 1):
            try:
                # GPT-5.1 Responses API 시도
                try:
                    response = self.client.responses.create(
                        model=self.model,
                        input=[
                            {"role": "developer", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        reasoning={"effort": self.reasoning_effort},
                        text={"verbosity": self.verbosity},
                        max_output_tokens=2000
                    )
                    result = response.output_text
                    if hasattr(response, 'usage'):
                        self.stats["tokens"] += response.usage.total_tokens

                except Exception as e:
                    # GPT-5.1 실패 시 Chat Completions API로 폴백
                    print(f"  GPT-5.1 실패, {self.fallback_model}로 폴백: {e}")
                    response = self.client.chat.completions.create(
                        model=self.fallback_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=2000,
                        temperature=0.7
                    )
                    result = response.choices[0].message.content
                    if hasattr(response, 'usage'):
                        self.stats["tokens"] += response.usage.total_tokens

                self.stats["success"] += 1
                return result

            except Exception as e:
                print(f"  API 호출 실패 (시도 {attempt + 1}/{retries + 1}): {e}")
                if attempt < retries:
                    time.sleep(2 ** attempt)

        self.stats["failed"] += 1
        return None

    def parse_json(self, response: str) -> Optional[Dict]:
        """JSON 파싱"""
        if not response:
            return None

        try:
            return json.loads(response.strip())
        except:
            pass

        # 마크다운 코드블록 제거
        try:
            if "```" in response:
                start = response.find("```json") or response.find("```")
                start = response.find("\n", start) + 1
                end = response.rfind("```")
                return json.loads(response[start:end].strip())
        except:
            pass

        # JSON 객체 추출
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except:
            pass

        return None


def generate_questions_with_api(
    selector: MultihopDocumentSelector,
    generator: GPT51QuestionGenerator,
    topic: str,
    count: int = 10
) -> List[Dict[str, Any]]:
    """
    GPT-5.1 API로 질문 생성

    Args:
        selector: Multi-hop 문서 선정기
        generator: GPT-5.1 생성기
        topic: 토픽
        count: 생성할 질문 수

    Returns:
        생성된 QA 리스트
    """
    print(f"\n=== GPT-5.1 질문 생성 ({topic}, {count}개) ===")
    generated = []

    # 타겟 문서 로드
    targets = [t for t in selector.targets if t["topic"] == topic][:count]

    # 질문 유형 분배 (10개 기준)
    # simple_factoid: 2, constraint: 2, multi_hop_2: 4, multi_hop_3: 1, reasoning: 1
    type_counts = {
        "simple_factoid": 2,
        "constraint": 2,
        "multi_hop_2": 4,
        "multi_hop_3": 1,
        "reasoning": 1
    }

    # 2-hop, 3-hop 문서 쌍 선정
    pairs = selector.select_2hop_pairs(topic=topic, count_per_target=1)
    triples = selector.select_3hop_triples(topic=topic, count_per_target=1)

    idx = 0

    # 1. Simple Factoid (2개)
    print("\n[Simple Factoid]")
    for i in range(min(type_counts["simple_factoid"], len(targets))):
        target = targets[i]
        prompt = PROMPTS["simple_factoid"].format(context=target["context"])

        print(f"  생성 중... ({i+1}/{type_counts['simple_factoid']})")
        response = generator.generate(SYSTEM_PROMPT, prompt)
        qa_data = generator.parse_json(response)

        if qa_data:
            qa_data["retrieval_gt"] = [target["doc_id"]]
            qa_data["target_doc_id"] = target["doc_id"]
            qa_data["topic"] = topic
            qa_data["question_type"] = "simple_factoid"
            generated.append(qa_data)
            print(f"    ✓ {qa_data.get('question', '')[:50]}...")
        else:
            print(f"    ✗ 파싱 실패")

    # 2. Constraint (2개)
    print("\n[Constraint]")
    for i in range(min(type_counts["constraint"], len(targets))):
        target = targets[i + 2] if i + 2 < len(targets) else targets[i]
        prompt = PROMPTS["constraint"].format(context=target["context"])

        print(f"  생성 중... ({i+1}/{type_counts['constraint']})")
        response = generator.generate(SYSTEM_PROMPT, prompt)
        qa_data = generator.parse_json(response)

        if qa_data:
            qa_data["retrieval_gt"] = [target["doc_id"]]
            qa_data["target_doc_id"] = target["doc_id"]
            qa_data["topic"] = topic
            qa_data["question_type"] = "constraint"
            generated.append(qa_data)
            print(f"    ✓ {qa_data.get('question', '')[:50]}...")
        else:
            print(f"    ✗ 파싱 실패")

    # 3. Multi-hop 2-hop (4개)
    print("\n[Multi-hop 2-hop]")
    for i in range(min(type_counts["multi_hop_2"], len(pairs))):
        pair = pairs[i]
        prompt = PROMPTS["multi_hop_2"].format(
            context_1=pair.target_context,
            context_2=pair.related_context
        )

        print(f"  생성 중... ({i+1}/{type_counts['multi_hop_2']})")
        response = generator.generate(SYSTEM_PROMPT_MULTIHOP, prompt)
        qa_data = generator.parse_json(response)

        if qa_data:
            qa_data["retrieval_gt"] = [pair.target_doc_id, pair.related_doc_id]
            qa_data["target_doc_id"] = pair.target_doc_id
            qa_data["related_doc_ids"] = [pair.related_doc_id]
            qa_data["topic"] = topic
            qa_data["question_type"] = "multi_hop_2"
            qa_data["similarity_score"] = pair.similarity_score
            generated.append(qa_data)
            print(f"    ✓ {qa_data.get('question', '')[:50]}...")
        else:
            print(f"    ✗ 파싱 실패")

    # 4. Multi-hop 3-hop (1개)
    print("\n[Multi-hop 3-hop]")
    if triples:
        triple = triples[0]
        prompt = PROMPTS["multi_hop_3"].format(
            context_1=triple.doc1_context,
            context_2=triple.doc2_context,
            context_3=triple.doc3_context
        )

        print(f"  생성 중... (1/{type_counts['multi_hop_3']})")
        response = generator.generate(SYSTEM_PROMPT_MULTIHOP, prompt)
        qa_data = generator.parse_json(response)

        if qa_data:
            qa_data["retrieval_gt"] = [triple.doc1_id, triple.doc2_id, triple.doc3_id]
            qa_data["target_doc_id"] = triple.doc1_id
            qa_data["related_doc_ids"] = [triple.doc2_id, triple.doc3_id]
            qa_data["topic"] = topic
            qa_data["question_type"] = "multi_hop_3"
            generated.append(qa_data)
            print(f"    ✓ {qa_data.get('question', '')[:50]}...")
        else:
            print(f"    ✗ 파싱 실패")

    # 5. Reasoning (1개)
    print("\n[Reasoning]")
    if len(targets) > 4:
        target = targets[4]
        prompt = PROMPTS["reasoning"].format(context=target["context"])

        print(f"  생성 중... (1/{type_counts['reasoning']})")
        response = generator.generate(SYSTEM_PROMPT, prompt)
        qa_data = generator.parse_json(response)

        if qa_data:
            qa_data["retrieval_gt"] = [target["doc_id"]]
            qa_data["target_doc_id"] = target["doc_id"]
            qa_data["topic"] = topic
            qa_data["question_type"] = "reasoning"
            generated.append(qa_data)
            print(f"    ✓ {qa_data.get('question', '')[:50]}...")
        else:
            print(f"    ✗ 파싱 실패")

    print(f"\n총 생성: {len(generated)}개")
    print(f"API 통계: {generator.stats}")

    return generated


def prepare_autorag_data(
    generated_qa: List[Dict],
    selector: MultihopDocumentSelector,
    output_dir: Path
) -> tuple:
    """
    AutoRAG 포맷 데이터 준비

    Returns:
        (qa_df, corpus_df)
    """
    print("\n=== AutoRAG 포맷 변환 ===")

    # QA 변환
    converter = AutoRAGConverter()
    autorag_qa = converter.convert_qa_list(generated_qa, id_prefix="e2e_test")

    qa_df = converter.to_dataframe(autorag_qa)
    print(f"QA: {len(qa_df)}개")

    # Corpus 준비 (retrieval_gt에 포함된 모든 문서)
    corpus_data = []
    seen_ids = set()

    # 타겟 문서들
    for target in selector.targets:
        if target["doc_id"] not in seen_ids:
            corpus_data.append({
                "doc_id": target["doc_id"],
                "contents": target["context"],
                "metadata": {
                    "topic": target["topic"],
                    "doc_title": target["doc_title"]
                }
            })
            seen_ids.add(target["doc_id"])

        # 관련 문서들
        for rd in target.get("related_documents", []):
            if rd["doc_id"] not in seen_ids:
                corpus_data.append({
                    "doc_id": rd["doc_id"],
                    "contents": rd["context"],
                    "metadata": {
                        "doc_title": rd["doc_title"]
                    }
                })
                seen_ids.add(rd["doc_id"])

    corpus_df = pd.DataFrame(corpus_data)
    print(f"Corpus: {len(corpus_df)}개")

    # 저장
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_path = output_dir / "qa.parquet"
    corpus_path = output_dir / "corpus.parquet"

    qa_df.to_parquet(qa_path, index=False)
    corpus_df.to_parquet(corpus_path, index=False)

    print(f"\n저장 완료:")
    print(f"  - {qa_path}")
    print(f"  - {corpus_path}")

    return qa_df, corpus_df


def run_e2e_test():
    """E2E 테스트 실행"""
    print("=" * 70)
    print("E2E 테스트: GPT-5.1 질문 생성 → AutoRAG 파이프라인 통합")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "output" / f"e2e_test_{timestamp}"

    # 1. Golden Dataset 로드 및 문서 선정기 초기화
    print("\n[1/4] Golden Dataset 로드...")
    base_path = Path(__file__).parent.parent / "golden_dataset" / "output"
    selector = MultihopDocumentSelector(str(base_path / "golden_dataset_v1.json"))

    # 2. GPT-5.1 질문 생성기 초기화
    print("\n[2/4] GPT-5.1 질문 생성기 초기화...")
    if not HAS_OPENAI:
        print("Error: openai 패키지가 필요합니다")
        return

    generator = GPT51QuestionGenerator(
        model="gpt-5.1",
        reasoning_effort="medium",
        verbosity="medium",
        fallback_model="gpt-4o-mini"
    )

    # 3. 질문 생성 (10개)
    print("\n[3/4] 질문 생성 (10개)...")
    topic = "공공행정"  # 첫 토픽
    generated_qa = generate_questions_with_api(
        selector=selector,
        generator=generator,
        topic=topic,
        count=10
    )

    if not generated_qa:
        print("Error: 질문 생성 실패")
        return

    # 생성 결과 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "generated_qa.json", 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "topic": topic,
                "model": generator.model,
                "stats": generator.stats
            },
            "qa_pairs": generated_qa
        }, f, ensure_ascii=False, indent=2)

    # 4. AutoRAG 포맷 변환
    print("\n[4/4] AutoRAG 포맷 변환...")
    qa_df, corpus_df = prepare_autorag_data(generated_qa, selector, output_dir)

    # 결과 요약
    print("\n" + "=" * 70)
    print("E2E 테스트 완료!")
    print("=" * 70)

    print(f"\n생성된 질문 요약:")
    type_counts = {}
    for qa in generated_qa:
        qt = qa.get("question_type", "unknown")
        type_counts[qt] = type_counts.get(qt, 0) + 1

    for qt, cnt in type_counts.items():
        print(f"  - {qt}: {cnt}개")

    print(f"\nMulti-hop 질문:")
    for qa in generated_qa:
        if qa.get("question_type", "").startswith("multi_hop"):
            print(f"  - [{qa['question_type']}] {qa.get('question', '')[:60]}...")
            print(f"    retrieval_gt: {qa.get('retrieval_gt', [])}")

    print(f"\n출력 디렉토리: {output_dir}")
    print(f"\n다음 단계:")
    print(f"  1. qa.parquet 검토: {output_dir / 'qa.parquet'}")
    print(f"  2. AutoRAG 평가 실행:")
    print(f"     autorag evaluate --qa_data_path {output_dir / 'qa.parquet'} ...")

    return {
        "output_dir": str(output_dir),
        "qa_count": len(generated_qa),
        "corpus_count": len(corpus_df),
        "api_stats": generator.stats
    }


if __name__ == "__main__":
    result = run_e2e_test()
    if result:
        print(f"\n최종 결과: {result}")
