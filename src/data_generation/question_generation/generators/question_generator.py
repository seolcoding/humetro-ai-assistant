"""
GPT-5.1 기반 질문 생성기

모델 선택 근거: MODEL_SELECTION.md 참조
- Model: gpt-5.1
- Reasoning Effort: medium (기본), high (5-hop)
- Verbosity: medium (필수 필드 + 추론 단계 포함)
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# 로깅 설정
def setup_logger(log_dir: str = None) -> logging.Logger:
    """로거 설정 - 콘솔 + 파일 출력"""
    logger = logging.getLogger("question_generator")
    logger.setLevel(logging.DEBUG)

    # 이미 핸들러가 있으면 초기화
    if logger.handlers:
        logger.handlers.clear()

    # 포맷 설정
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (log_dir이 지정된 경우)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"로그 파일: {log_file}")

    return logger

# 기본 로거
logger = setup_logger()

try:
    from openai import OpenAI
except ImportError:
    print("openai 패키지가 필요합니다: pip install openai")
    OpenAI = None

# 프롬프트 임포트
import sys
sys.path.append(str(Path(__file__).parent.parent))
from prompts.system_prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_MULTIHOP
from prompts.question_prompts import PROMPTS


@dataclass
class GeneratedQA:
    """생성된 QA 데이터 구조"""
    question: str
    answer: str
    question_type: str
    evidence: Optional[str] = None
    constraint: Optional[str] = None
    reasoning_steps: Optional[List[str]] = None
    reasoning_type: Optional[str] = None
    structure_type: Optional[str] = None  # linear | branching (5-hop용)

    # 문서별 근거
    evidence_doc1: Optional[str] = None
    evidence_doc2: Optional[str] = None
    evidence_doc3: Optional[str] = None
    evidence_doc4: Optional[str] = None
    evidence_doc5: Optional[str] = None
    evidence_docs: Optional[List[str]] = None  # Multi-doc용

    # 문서 메타데이터
    retrieval_gt: Optional[List[str]] = None
    target_doc_id: Optional[str] = None
    related_doc_ids: Optional[List[str]] = None
    doc_usage: Optional[str] = None  # Multi-doc용

    # 기타 메타데이터
    topic: Optional[str] = None
    generation_model: str = "gpt-5.1"


class QuestionGenerator:
    """
    GPT-5.1 기반 질문 생성기

    모델 선택 근거 (MODEL_SELECTION.md):
    - GPT-5.1: 최신 flagship 모델, SWE-bench 76.3%, 2-3x faster
    - Reasoning Effort: medium - Multi-hop 추론 경로 설계에 적합
    - Structured Outputs: JSON Schema 강제로 파싱 실패 방지
    """

    # 질문 유형별 JSON 스키마 정의 (question + answer + evidence)
    SCHEMAS = {
        "simple_factoid": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "생성된 질문"},
                "answer": {"type": "string", "description": "문서에서 추출한 답변"},
                "evidence": {"type": "string", "description": "답변 근거 문장"}
            },
            "required": ["question", "answer", "evidence"],
            "additionalProperties": False
        },
        "constraint": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "answer": {"type": "string"},
                "evidence": {"type": "string"}
            },
            "required": ["question", "answer", "evidence"],
            "additionalProperties": False
        },
        "reasoning": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "answer": {"type": "string"},
                "evidence": {"type": "string"}
            },
            "required": ["question", "answer", "evidence"],
            "additionalProperties": False
        },
        "multi_doc_1": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "answer": {"type": "string"},
                "evidence": {"type": "string", "description": "종합 근거"}
            },
            "required": ["question", "answer", "evidence"],
            "additionalProperties": False
        },
        "multi_hop_2": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "answer": {"type": "string"},
                "evidence": {"type": "string", "description": "추론 경로 요약"}
            },
            "required": ["question", "answer", "evidence"],
            "additionalProperties": False
        },
        "multi_hop_3": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "answer": {"type": "string"},
                "evidence": {"type": "string", "description": "추론 경로 요약"}
            },
            "required": ["question", "answer", "evidence"],
            "additionalProperties": False
        },
        "multi_hop_5": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "answer": {"type": "string"},
                "evidence": {"type": "string", "description": "추론 경로 요약"}
            },
            "required": ["question", "answer", "evidence"],
            "additionalProperties": False
        }
    }

    def __init__(
        self,
        model: str = "gpt-5.1",
        reasoning_effort: str = "medium",
        max_output_tokens: int = 20000,
        api_key: Optional[str] = None,
        use_structured_output: bool = True
    ):
        """
        Args:
            model: OpenAI 모델명 (기본: gpt-5.1)
            reasoning_effort: 추론 강도 (none/low/medium/high)
            max_output_tokens: 최대 출력 토큰
            api_key: OpenAI API 키 (없으면 환경변수 사용)
            use_structured_output: Structured Outputs 사용 여부 (권장: True)
        """
        if OpenAI is None:
            raise ImportError("openai 패키지를 설치하세요: pip install openai")

        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.use_structured_output = use_structured_output

        # API 클라이언트 초기화 (5-hop은 추론에 5분 이상 걸릴 수 있음)
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            timeout=600.0  # 10분 타임아웃
        )

        # 통계
        self.stats = {
            "total_calls": 0,
            "successful": 0,
            "failed": 0,
            "total_tokens": 0
        }

    def generate_single_doc_question(
        self,
        context: str,
        question_type: str,
        doc_id: str,
        topic: str
    ) -> Optional[GeneratedQA]:
        """
        단일 문서 기반 질문 생성 (Simple Factoid, Constraint, Reasoning)

        Args:
            context: 문서 내용
            question_type: simple_factoid | constraint | reasoning
            doc_id: 문서 ID
            topic: 토픽

        Returns:
            GeneratedQA 객체 또는 None (실패 시)
        """
        if question_type not in ["simple_factoid", "constraint", "reasoning"]:
            raise ValueError(f"Invalid question_type for single doc: {question_type}")

        # 프롬프트 구성
        prompt = PROMPTS[question_type].format(context=context)

        # API 호출 (Structured Outputs 적용)
        response = self._call_api(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            question_type=question_type
        )

        if response is None:
            return None

        # JSON 파싱
        qa_data = self._parse_json_response(response)
        if qa_data is None:
            return None

        # GeneratedQA 객체 생성
        qa = GeneratedQA(
            question=qa_data.get("question", ""),
            answer=qa_data.get("answer", ""),
            question_type=question_type,
            evidence=qa_data.get("evidence"),
            constraint=qa_data.get("constraint"),
            reasoning_type=qa_data.get("reasoning_type"),
            retrieval_gt=[doc_id],
            target_doc_id=doc_id,
            topic=topic,
            generation_model=self.model
        )

        return qa

    def generate_multihop_2_question(
        self,
        context_1: str,
        context_2: str,
        doc_id_1: str,
        doc_id_2: str,
        topic: str
    ) -> Optional[GeneratedQA]:
        """
        2-hop Multi-hop 질문 생성

        Args:
            context_1: 시작 문서 내용
            context_2: 연결 문서 내용
            doc_id_1: 시작 문서 ID
            doc_id_2: 연결 문서 ID
            topic: 토픽

        Returns:
            GeneratedQA 객체 또는 None
        """
        # 프롬프트 구성
        prompt = PROMPTS["multi_hop_2"].format(
            context_1=context_1,
            context_2=context_2
        )

        # API 호출 (Structured Outputs 적용)
        response = self._call_api(
            system_prompt=SYSTEM_PROMPT_MULTIHOP,
            user_prompt=prompt,
            question_type="multi_hop_2"
        )

        if response is None:
            return None

        # JSON 파싱
        qa_data = self._parse_json_response(response)
        if qa_data is None:
            return None

        # GeneratedQA 객체 생성
        qa = GeneratedQA(
            question=qa_data.get("question", ""),
            answer=qa_data.get("answer", ""),
            question_type="multi_hop_2",
            reasoning_steps=qa_data.get("reasoning_steps"),
            evidence_doc1=qa_data.get("evidence_doc1"),
            evidence_doc2=qa_data.get("evidence_doc2"),
            retrieval_gt=[doc_id_1, doc_id_2],
            target_doc_id=doc_id_1,
            related_doc_ids=[doc_id_2],
            topic=topic,
            generation_model=self.model
        )

        return qa

    def generate_multihop_3_question(
        self,
        context_1: str,
        context_2: str,
        context_3: str,
        doc_id_1: str,
        doc_id_2: str,
        doc_id_3: str,
        topic: str
    ) -> Optional[GeneratedQA]:
        """
        3-hop Multi-hop 질문 생성

        Args:
            context_1: 시작 문서 내용
            context_2: 중간 문서 내용
            context_3: 최종 문서 내용
            doc_id_1, doc_id_2, doc_id_3: 문서 IDs
            topic: 토픽

        Returns:
            GeneratedQA 객체 또는 None
        """
        # 프롬프트 구성
        prompt = PROMPTS["multi_hop_3"].format(
            context_1=context_1,
            context_2=context_2,
            context_3=context_3
        )

        # API 호출 (Structured Outputs 적용)
        response = self._call_api(
            system_prompt=SYSTEM_PROMPT_MULTIHOP,
            user_prompt=prompt,
            question_type="multi_hop_3"
        )

        if response is None:
            return None

        # JSON 파싱
        qa_data = self._parse_json_response(response)
        if qa_data is None:
            return None

        # GeneratedQA 객체 생성
        qa = GeneratedQA(
            question=qa_data.get("question", ""),
            answer=qa_data.get("answer", ""),
            question_type="multi_hop_3",
            reasoning_steps=qa_data.get("reasoning_steps"),
            evidence_doc1=qa_data.get("evidence_doc1"),
            evidence_doc2=qa_data.get("evidence_doc2"),
            evidence_doc3=qa_data.get("evidence_doc3"),
            retrieval_gt=[doc_id_1, doc_id_2, doc_id_3],
            target_doc_id=doc_id_1,
            related_doc_ids=[doc_id_2, doc_id_3],
            topic=topic,
            generation_model=self.model
        )

        return qa

    def generate_multidoc_1_question(
        self,
        contexts: List[str],
        doc_ids: List[str],
        topic: str
    ) -> Optional[GeneratedQA]:
        """
        Multi-doc 1-hop 질문 생성 (비교/종합)

        Args:
            contexts: 문서 내용 리스트 (2-4개)
            doc_ids: 문서 ID 리스트
            topic: 토픽

        Returns:
            GeneratedQA 객체 또는 None
        """
        # 문서들 포맷팅
        contexts_str = "\n\n".join([
            f"### 문서 {i+1}\n{ctx}"
            for i, ctx in enumerate(contexts)
        ])

        prompt = PROMPTS["multi_doc_1"].format(contexts=contexts_str)

        # API 호출 (Structured Outputs 적용)
        response = self._call_api(
            system_prompt=SYSTEM_PROMPT_MULTIHOP,
            user_prompt=prompt,
            question_type="multi_doc_1"
        )

        if response is None:
            return None

        qa_data = self._parse_json_response(response)
        if qa_data is None:
            return None

        qa = GeneratedQA(
            question=qa_data.get("question", ""),
            answer=qa_data.get("answer", ""),
            question_type="multi_doc_1",
            doc_usage=qa_data.get("doc_usage"),
            evidence_docs=qa_data.get("evidence_docs"),
            retrieval_gt=doc_ids,
            target_doc_id=doc_ids[0],
            related_doc_ids=doc_ids[1:],
            topic=topic,
            generation_model=self.model
        )

        return qa

    def generate_multihop_5_question(
        self,
        context_1: str,
        context_2: str,
        context_3: str,
        context_4: str,
        context_5: str,
        doc_id_1: str,
        doc_id_2: str,
        doc_id_3: str,
        doc_id_4: str,
        doc_id_5: str,
        topic: str
    ) -> Optional[GeneratedQA]:
        """
        5-hop Multi-hop 질문 생성 (선형/비선형)

        Args:
            context_1~5: 문서 내용들
            doc_id_1~5: 문서 IDs
            topic: 토픽

        Returns:
            GeneratedQA 객체 또는 None
        """
        prompt = PROMPTS["multi_hop_5"].format(
            context_1=context_1,
            context_2=context_2,
            context_3=context_3,
            context_4=context_4,
            context_5=context_5
        )

        # API 호출 (Structured Outputs 적용)
        response = self._call_api(
            system_prompt=SYSTEM_PROMPT_MULTIHOP,
            user_prompt=prompt,
            question_type="multi_hop_5"
        )

        if response is None:
            return None

        qa_data = self._parse_json_response(response)
        if qa_data is None:
            return None

        qa = GeneratedQA(
            question=qa_data.get("question", ""),
            answer=qa_data.get("answer", ""),
            question_type="multi_hop_5",
            structure_type=qa_data.get("structure_type", "linear"),
            reasoning_steps=qa_data.get("reasoning_steps"),
            evidence_doc1=qa_data.get("evidence_doc1"),
            evidence_doc2=qa_data.get("evidence_doc2"),
            evidence_doc3=qa_data.get("evidence_doc3"),
            evidence_doc4=qa_data.get("evidence_doc4"),
            evidence_doc5=qa_data.get("evidence_doc5"),
            retrieval_gt=[doc_id_1, doc_id_2, doc_id_3, doc_id_4, doc_id_5],
            target_doc_id=doc_id_1,
            related_doc_ids=[doc_id_2, doc_id_3, doc_id_4, doc_id_5],
            topic=topic,
            generation_model=self.model
        )

        return qa

    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        question_type: str,
        retries: int = 3
    ) -> Optional[str]:
        """
        GPT-5.1 API 호출 (Structured Outputs 지원)

        GPT-5.1 Responses API 사용:
        - reasoning.effort: medium (기본), high (5-hop)
        - text.format: json_schema (출력 형식 강제)

        Args:
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            question_type: 질문 유형 (스키마 선택용)
            retries: 재시도 횟수

        Returns:
            응답 텍스트 또는 None
        """
        self.stats["total_calls"] += 1

        # 질문 유형에 맞는 스키마 선택
        schema = self.SCHEMAS.get(question_type)

        # 5-hop도 medium으로 설정 (high는 5개 문서에서 타임아웃 발생)
        # TODO: high가 필요하면 문서 요약본 사용 검토
        reasoning_effort = self.reasoning_effort  # 모두 medium 사용

        # 입력 프롬프트 길이 계산 (대략적)
        input_chars = len(system_prompt) + len(user_prompt)
        logger.debug(f"[API] type={question_type}, effort={reasoning_effort}, input_chars={input_chars}")

        for attempt in range(retries):
            try:
                start_time = time.time()
                logger.debug(f"[API] 시도 {attempt + 1}/{retries} 시작...")

                # GPT-5.1 Responses API 호출
                # 참고: https://platform.openai.com/docs/api-reference/responses
                api_params = {
                    "model": self.model,
                    "input": [
                        {"role": "developer", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "reasoning": {"effort": reasoning_effort},
                    "max_output_tokens": self.max_output_tokens
                }

                # Structured Outputs 적용 (스키마가 있고 옵션이 켜진 경우)
                if self.use_structured_output and schema:
                    api_params["text"] = {
                        "format": {
                            "type": "json_schema",
                            "name": f"qa_{question_type}",
                            "schema": schema,
                            "strict": True
                        }
                    }

                response = self.client.responses.create(**api_params)
                elapsed = time.time() - start_time

                # 토큰 사용량 기록
                tokens_used = 0
                reasoning_tokens = 0
                if hasattr(response, 'usage') and response.usage:
                    tokens_used = response.usage.total_tokens
                    self.stats["total_tokens"] += tokens_used
                    if hasattr(response.usage, 'output_tokens_details'):
                        reasoning_tokens = getattr(response.usage.output_tokens_details, 'reasoning_tokens', 0)

                self.stats["successful"] += 1

                # output_text가 비어있으면 output에서 텍스트 추출 시도
                result = response.output_text if response.output_text else ""
                if not result and hasattr(response, 'output') and response.output is not None:
                    for item in response.output:
                        if item is None:
                            continue
                        if hasattr(item, 'content') and item.content is not None:
                            for content in item.content:
                                if content is None:
                                    continue
                                if hasattr(content, 'text') and content.text:
                                    result = content.text
                                    break
                        if result:
                            break

                # 응답이 비어있으면 에러 처리
                if not result:
                    # 디버깅: 응답 객체 상세 로깅
                    logger.warning(f"[API] 빈 응답 - type={question_type}, elapsed={elapsed:.1f}s")
                    logger.debug(f"[API] response.output_text: {repr(response.output_text)}")
                    logger.debug(f"[API] response.output type: {type(response.output)}")
                    if hasattr(response, 'output') and response.output:
                        for i, item in enumerate(response.output):
                            logger.debug(f"[API] output[{i}] type: {type(item)}, value: {item}")
                    # 응답에 stop_reason 확인
                    if hasattr(response, 'stop_reason'):
                        logger.warning(f"[API] stop_reason: {response.stop_reason}")
                    if hasattr(response, 'status'):
                        logger.warning(f"[API] status: {response.status}")
                    raise ValueError("Empty response from API")

                # 성공 로그
                logger.info(f"[API] ✓ {question_type} 완료 | {elapsed:.1f}s | {tokens_used} tokens (reasoning: {reasoning_tokens})")
                logger.debug(f"[API] 응답 미리보기: {result[:200]}...")

                return result

            except Exception as e:
                logger.warning(f"[API] ✗ 실패 (시도 {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    logger.debug(f"[API] {wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    self.stats["failed"] += 1
                    logger.error(f"[API] 최종 실패: {question_type}")
                    return None

        return None

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        API 응답에서 JSON 파싱

        GPT-5.1은 JSON 출력이 안정적이지만, 예외 처리 포함

        Args:
            response: API 응답 텍스트

        Returns:
            파싱된 딕셔너리 또는 None
        """
        def ensure_dict(data):
            """리스트인 경우 첫 번째 요소 반환"""
            if isinstance(data, list):
                return data[0] if data else None
            return data

        try:
            # 순수 JSON 파싱 시도
            parsed = json.loads(response.strip())
            return ensure_dict(parsed)
        except json.JSONDecodeError:
            pass

        # 마크다운 코드블록 제거 시도
        try:
            # ```json ... ``` 형태 처리
            if "```" in response:
                start = response.find("```json")
                if start == -1:
                    start = response.find("```")
                start = response.find("\n", start) + 1
                end = response.rfind("```")
                json_str = response[start:end].strip()
                parsed = json.loads(json_str)
                return ensure_dict(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

        # JSON 객체 추출 시도 (중첩 브래킷 처리)
        try:
            start = response.find("{")
            if start != -1:
                # 균형 맞춘 JSON 추출
                depth = 0
                in_string = False
                escape = False
                end = start

                for i, char in enumerate(response[start:], start):
                    if escape:
                        escape = False
                        continue
                    if char == '\\' and in_string:
                        escape = True
                        continue
                    if char == '"' and not escape:
                        in_string = not in_string
                        continue
                    if not in_string:
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break

                if end > start:
                    json_str = response[start:end]
                    parsed = json.loads(json_str)
                    return ensure_dict(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

        # 필수 필드 추출 (마지막 수단)
        try:
            question_match = re.search(r'"question"\s*:\s*"([^"]+(?:\\.[^"]*)*)"', response)
            answer_match = re.search(r'"answer"\s*:\s*"([^"]+(?:\\.[^"]*)*)"', response)

            if question_match and answer_match:
                result = {
                    "question": question_match.group(1).replace('\\"', '"'),
                    "answer": answer_match.group(1).replace('\\"', '"')
                }
                # 추가 필드 추출
                for field in ["evidence", "doc_usage", "structure_type"]:
                    field_match = re.search(f'"{field}"\\s*:\\s*"([^"]+(?:\\\\.[^"]*)*)"', response)
                    if field_match:
                        result[field] = field_match.group(1).replace('\\"', '"')
                return result
        except:
            pass

        # 디버그: 파싱 실패 시 전체 응답 확인
        print(f"JSON 파싱 실패: {response[:500]}...")
        print(f"응답 길이: {len(response)}, 첫 문자: '{response[0] if response else 'N/A'}', 마지막 문자: '{response[-1] if response else 'N/A'}'")
        return None

    def get_stats(self) -> Dict[str, int]:
        """통계 반환"""
        return self.stats.copy()


# 테스트용 Mock 생성기 (API 키 없을 때)
class MockQuestionGenerator(QuestionGenerator):
    """테스트용 Mock 생성기 (실제 API 호출 없음)"""

    def __init__(self, **kwargs):
        self.model = kwargs.get("model", "gpt-5.1-mock")
        self.stats = {"total_calls": 0, "successful": 0, "failed": 0, "total_tokens": 0}

    def _call_api(self, system_prompt: str, user_prompt: str, retries: int = 3) -> str:
        """Mock API 응답"""
        self.stats["total_calls"] += 1
        self.stats["successful"] += 1

        # 질문 유형 감지
        if "simple_factoid" in user_prompt.lower():
            return json.dumps({
                "question": "[Mock] 테스트 질문입니다",
                "answer": "테스트 답변",
                "evidence": "테스트 근거",
                "question_type": "simple_factoid"
            }, ensure_ascii=False)
        elif "multi_hop_2" in user_prompt.lower() or "2-hop" in user_prompt:
            return json.dumps({
                "question": "[Mock] 2-hop 테스트 질문",
                "answer": "테스트 답변",
                "reasoning_steps": ["Step 1: 테스트", "Step 2: 테스트"],
                "evidence_doc1": "근거 1",
                "evidence_doc2": "근거 2",
                "question_type": "multi_hop_2"
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "question": "[Mock] 기본 테스트 질문",
                "answer": "테스트 답변",
                "evidence": "테스트 근거",
                "question_type": "unknown"
            }, ensure_ascii=False)


if __name__ == "__main__":
    # 간단한 테스트
    print("=== QuestionGenerator 테스트 ===")

    # Mock 생성기로 테스트
    generator = MockQuestionGenerator()

    # Simple Factoid 테스트
    qa = generator.generate_single_doc_question(
        context="2024년 강남구 청소년 예산은 52억원이다.",
        question_type="simple_factoid",
        doc_id="test_001",
        topic="공공행정"
    )

    if qa:
        print(f"생성된 질문: {qa.question}")
        print(f"답변: {qa.answer}")
        print(f"유형: {qa.question_type}")

    print(f"\n통계: {generator.get_stats()}")
