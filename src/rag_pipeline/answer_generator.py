#!/usr/bin/env python3
"""
Answer Generation Module
=========================

답변 생성과 평가를 분리
- Phase 1: Answer Generation (이 모듈)
- Phase 2: Parallel Evaluation (parallel_evaluator.py)

이 모듈은 모든 모델의 답변을 생성하고 저장
평가는 별도의 병렬 프로세스에서 수행
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pickle
from tqdm import tqdm
import litellm

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    모델 답변 생성기

    주어진 질문과 컨텍스트에 대해 답변 생성
    평가와 독립적으로 동작
    """

    def __init__(self, retriever: Optional[Any] = None, k_documents: int = 4):
        """
        Args:
            retriever: LangChain retriever (optional)
            k_documents: 문서 검색 개수
        """
        self.retriever = retriever
        self.k_documents = k_documents

        # Configure LiteLLM
        litellm.drop_params = True
        litellm.set_verbose = False

    def generate_answers_for_model(
        self,
        model_config: Dict[str, str],
        questions: List[dict],
        fixed_contexts: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        단일 모델에 대한 모든 답변 생성

        Args:
            model_config: 모델 설정
                {"name": "GPT-4o-mini", "model": "gpt-4o-mini", "api_base": None}
            questions: 질문 리스트
            fixed_contexts: 사전 검색된 컨텍스트 (optional)

        Returns:
            RAGAS 평가용 데이터셋 형식
            {
                "user_input": ["Q1", "Q2", ...],
                "response": ["A1", "A2", ...],
                "reference": ["GT1", "GT2", ...],
                "retrieved_contexts": [[C1, C2], [C1, C2], ...]
            }
        """
        model_name = model_config["name"]
        logger.info(f"🤖 {model_name} 답변 생성 시작...")

        user_inputs = []
        responses = []
        references = []
        retrieved_contexts = []

        for q in tqdm(questions, desc=f"{model_name} 답변 생성"):
            question_text = q.get("question", q.get("user_input", ""))
            reference_answer = q.get("ground_truth", q.get("reference_answer", q.get("reference", "")))

            # Get context
            if fixed_contexts and question_text in fixed_contexts:
                contexts = fixed_contexts[question_text]
            else:
                contexts = self._retrieve_context(question_text)

            # Generate answer
            answer = self._generate_answer(
                model_config=model_config,
                question=question_text,
                contexts=contexts
            )

            user_inputs.append(question_text)
            responses.append(answer)
            references.append(reference_answer)
            retrieved_contexts.append(contexts)

        dataset = {
            "user_input": user_inputs,
            "response": responses,
            "reference": references,
            "retrieved_contexts": retrieved_contexts
        }

        logger.info(f"✅ {model_name} 답변 생성 완료 ({len(user_inputs)}개)")

        return dataset

    def generate_all_answers(
        self,
        models: List[Dict[str, str]],
        questions: List[dict],
        use_fixed_context: bool = True
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        모든 모델에 대한 답변 생성

        Args:
            models: 모델 설정 리스트
            questions: 질문 리스트
            use_fixed_context: 고정 컨텍스트 사용 여부

        Returns:
            {
                "GPT-4o-mini": {"user_input": [...], "response": [...], ...},
                "EXAONE-3.5": {...},
                ...
            }
        """
        logger.info("="*70)
        logger.info("Phase 1: 답변 생성")
        logger.info("="*70)

        # Prepare fixed contexts if needed
        fixed_contexts = {}
        if use_fixed_context and self.retriever:
            logger.info("고정 컨텍스트 준비 중...")
            for q in tqdm(questions, desc="컨텍스트 검색"):
                question_text = q.get("question", q.get("user_input", ""))
                contexts = self._retrieve_context(question_text)
                fixed_contexts[question_text] = contexts

        # Generate answers for each model
        all_datasets = {}
        for model_config in models:
            model_name = model_config["name"]

            try:
                # Test connection first
                if not self._test_model_connection(model_config):
                    logger.error(f"❌ {model_name} 연결 실패, 건너뜀")
                    continue

                # Generate answers
                dataset = self.generate_answers_for_model(
                    model_config=model_config,
                    questions=questions,
                    fixed_contexts=fixed_contexts if use_fixed_context else None
                )

                all_datasets[model_name] = dataset

            except Exception as e:
                logger.error(f"❌ {model_name} 답변 생성 실패: {e}")
                continue

        logger.info(f"\n✅ 답변 생성 완료: {len(all_datasets)}/{len(models)} 모델")

        return all_datasets

    def save_answers(
        self,
        all_datasets: Dict[str, Dict[str, List[str]]],
        output_dir: Path,
        experiment_id: int
    ):
        """답변 데이터 저장 (체크포인트)"""
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = output_dir / f"exp_{experiment_id}_answers.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(all_datasets, f)

        logger.info(f"💾 답변 저장: {checkpoint_file}")

    def load_answers(
        self,
        output_dir: Path,
        experiment_id: int
    ) -> Optional[Dict[str, Dict[str, List[str]]]]:
        """저장된 답변 로드"""
        checkpoint_file = output_dir / f"exp_{experiment_id}_answers.pkl"

        if not checkpoint_file.exists():
            return None

        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)

    def _retrieve_context(self, question: str) -> List[str]:
        """컨텍스트 검색"""
        if not self.retriever:
            return []

        try:
            documents = self.retriever.invoke(question)
            contexts = [doc.page_content for doc in documents]
            return contexts
        except Exception as e:
            logger.warning(f"⚠️ 컨텍스트 검색 실패: {e}")
            return []

    def _generate_answer(
        self,
        model_config: Dict[str, str],
        question: str,
        contexts: List[str]
    ) -> str:
        """답변 생성"""
        # Prepare context string
        context_str = "\n\n".join(contexts) if contexts else "관련 정보를 찾을 수 없습니다."

        # Create prompt
        prompt = f"""다음 컨텍스트를 바탕으로 질문에 답변해주세요.

컨텍스트:
{context_str}

질문: {question}

답변:"""

        try:
            # Get max_tokens from model config
            max_tokens = 2048
            if "config" in model_config:
                max_tokens = model_config["config"].get("max_tokens", 2048)

            response = litellm.completion(
                model=model_config["model"],
                messages=[
                    {"role": "system", "content": "당신은 서울 대중교통 정보를 제공하는 도우미입니다."},
                    {"role": "user", "content": prompt}
                ],
                api_base=model_config.get("api_base"),
                temperature=0.7,
                max_tokens=max_tokens,
                timeout=60
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            return f"Error: {str(e)}"

    def _test_model_connection(self, model_config: Dict[str, str]) -> bool:
        """모델 연결 테스트"""
        try:
            response = litellm.completion(
                model=model_config["model"],
                messages=[{"role": "user", "content": "Hello"}],
                api_base=model_config.get("api_base"),
                max_tokens=10,
                timeout=10
            )
            return True
        except Exception as e:
            logger.error(f"연결 테스트 실패: {e}")
            return False


def generate_answers_only(
    models: List[Dict[str, str]],
    questions: List[dict],
    retriever: Optional[Any] = None,
    use_fixed_context: bool = True,
    k_documents: int = 4,
    output_dir: Optional[Path] = None,
    experiment_id: Optional[int] = None
) -> Dict[str, Dict[str, List[str]]]:
    """
    답변만 생성 (평가 없이) - 메인 인터페이스

    Args:
        models: 모델 설정 리스트
        questions: 질문 리스트
        retriever: LangChain retriever
        use_fixed_context: 고정 컨텍스트 사용
        k_documents: 검색 문서 수
        output_dir: 저장 디렉토리
        experiment_id: 실험 ID

    Returns:
        모델별 답변 데이터셋
    """
    generator = AnswerGenerator(retriever=retriever, k_documents=k_documents)

    all_datasets = generator.generate_all_answers(
        models=models,
        questions=questions,
        use_fixed_context=use_fixed_context
    )

    if output_dir and experiment_id is not None:
        generator.save_answers(all_datasets, output_dir, experiment_id)

    return all_datasets


if __name__ == "__main__":
    # Test example
    logging.basicConfig(level=logging.INFO)

    test_models = [
        {"name": "GPT-4o-mini", "model": "gpt-4o-mini", "api_base": None}
    ]

    test_questions = [
        {"question": "What is RAG?", "ground_truth": "RAG is..."},
        {"question": "How does it work?", "ground_truth": "It works by..."}
    ]

    print("답변 생성 테스트...")
    datasets = generate_answers_only(
        models=test_models,
        questions=test_questions,
        use_fixed_context=False
    )

    print("\n결과:")
    for model_name, dataset in datasets.items():
        print(f"\n{model_name}:")
        print(f"  Questions: {len(dataset['user_input'])}")
        print(f"  Answers: {dataset['response'][:1]}")  # Show first answer
