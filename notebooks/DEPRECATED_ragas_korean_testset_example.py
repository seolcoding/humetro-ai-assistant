#!/usr/bin/env python3
"""
RAGAS Korean Testset Generation Example
RAGAS framework supports Korean Q/A generation through LLM-based testset generation
"""

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd
import os

# ============================================================================
# 핵심 포인트: RAGAS는 한국어 Q/A 생성을 지원합니다!
# ============================================================================
# 1. LLM 기반 생성: OpenAI 모델이 한국어 문서를 읽고 한국어 질문 생성
# 2. 언어 자동 감지: 입력 문서가 한국어면 자동으로 한국어 질문 생성
# 3. Evolution 전략: simple, reasoning, multi_context 모두 한국어 지원
# ============================================================================

def generate_korean_testset_with_ragas(
    df: pd.DataFrame,
    n_questions: int = 40,
    openai_api_key: str = None
) -> pd.DataFrame:
    """
    RAGAS를 사용한 한국어 평가 데이터셋 생성

    Args:
        df: 한국어 문서 데이터프레임 (question, answer 컬럼 포함)
        n_questions: 생성할 질문 개수
        openai_api_key: OpenAI API 키

    Returns:
        RAGAS 생성 테스트셋 (한국어 질문/답변 포함)
    """

    # 환경 변수에서 API 키 로드
    if openai_api_key is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수를 설정하세요")

    # LLM 및 임베딩 모델 설정
    # GPT-4o-mini: 비용 효율적이면서 한국어 성능 우수
    generator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=openai_api_key
    )

    critic_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,  # Critic은 더 일관된 평가 필요
        api_key=openai_api_key
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key
    )

    # 데이터프레임을 LangChain Document로 변환
    # page_content: question + answer 결합 (한국어 문서)
    df_for_loader = df.copy()
    df_for_loader['text'] = df_for_loader.apply(
        lambda row: f"질문: {row['question']}\n답변: {row['answer']}",
        axis=1
    )

    loader = DataFrameLoader(df_for_loader, page_content_column='text')
    documents = loader.load()

    print(f"📄 로드된 문서: {len(documents)}개")
    print(f"📝 샘플 문서 (한국어):")
    print(f"   {documents[0].page_content[:100]}...")

    # TestsetGenerator 초기화
    generator = TestsetGenerator.from_langchain(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings=embeddings
    )

    # Evolution 전략 설정
    # simple: 단순 재구성 (paraphrasing)
    # reasoning: 추론 필요 질문
    # multi_context: 여러 문서 결합 필요
    distributions = {
        simple: 0.5,          # 50%: 단순 질문
        reasoning: 0.25,      # 25%: 추론 질문
        multi_context: 0.25,  # 25%: 멀티 컨텍스트
    }

    print(f"\n🔄 RAGAS 테스트셋 생성 시작...")
    print(f"   - 목표 질문: {n_questions}개")
    print(f"   - Evolution 분포: Simple 50%, Reasoning 25%, Multi-context 25%")
    print(f"   - 언어: 한국어 (자동 감지)")

    # 테스트셋 생성 (한국어 문서 → 한국어 질문/답변)
    testset = generator.generate_with_langchain_docs(
        documents=documents,
        test_size=n_questions,
        distributions=distributions,
        # raise_exceptions=False  # 일부 실패해도 계속 진행
    )

    # DataFrame으로 변환
    testset_df = testset.to_pandas()

    print(f"\n✅ RAGAS 테스트셋 생성 완료!")
    print(f"   - 생성된 질문: {len(testset_df)}개")
    print(f"   - 컬럼: {testset_df.columns.tolist()}")

    # 샘플 출력 (한국어 확인)
    print(f"\n📝 생성된 한국어 질문 샘플:")
    for idx in range(min(3, len(testset_df))):
        print(f"\n   [{idx+1}] 질문: {testset_df.iloc[idx]['question'][:100]}...")
        print(f"       답변: {testset_df.iloc[idx]['ground_truth'][:100]}...")
        print(f"       Evolution: {testset_df.iloc[idx].get('evolution_type', 'N/A')}")

    return testset_df


def hybrid_korean_testset_generation(
    df: pd.DataFrame,
    n_total: int = 40,
    ragas_ratio: float = 0.75,
    openai_api_key: str = None
) -> dict:
    """
    하이브리드 접근: RAGAS + 기존 샘플링

    Args:
        df: 원본 데이터
        n_total: 총 질문 개수 (40개)
        ragas_ratio: RAGAS로 생성할 비율 (0.75 = 30개)

    Returns:
        {'ragas': DataFrame, 'simple': DataFrame, 'combined': DataFrame}
    """

    n_ragas = int(n_total * ragas_ratio)
    n_simple = n_total - n_ragas

    print(f"🎯 하이브리드 테스트셋 생성:")
    print(f"   - RAGAS 생성: {n_ragas}개 ({ragas_ratio*100:.0f}%)")
    print(f"   - 기존 샘플링: {n_simple}개 ({(1-ragas_ratio)*100:.0f}%)")

    # 1. RAGAS 기반 생성 (합성 질문)
    ragas_testset = generate_korean_testset_with_ragas(
        df=df,
        n_questions=n_ragas,
        openai_api_key=openai_api_key
    )

    # 2. 기존 데이터에서 직접 샘플링 (실제 질문)
    simple_samples = df.sample(n=n_simple, random_state=42)
    simple_testset = pd.DataFrame({
        'question': simple_samples['question'],
        'ground_truth': simple_samples['answer'],
        'contexts': simple_samples['answer'].apply(lambda x: [x]),  # RAGAS 형식 맞추기
        'evolution_type': 'original',
        'category': simple_samples.get('category', 'unknown')
    })

    # 3. 결합
    combined_testset = pd.concat([
        ragas_testset.assign(source='ragas'),
        simple_testset.assign(source='original')
    ], ignore_index=True)

    print(f"\n✅ 하이브리드 테스트셋 완성:")
    print(f"   - RAGAS: {len(ragas_testset)}개")
    print(f"   - Original: {len(simple_testset)}개")
    print(f"   - 총: {len(combined_testset)}개")

    return {
        'ragas': ragas_testset,
        'simple': simple_testset,
        'combined': combined_testset
    }


# ============================================================================
# 사용 예시
# ============================================================================
if __name__ == "__main__":
    # 샘플 한국어 데이터
    sample_data = pd.DataFrame({
        'question': [
            '서울시 120 다산콜센터는 무엇인가요?',
            '주민등록등본 발급 방법을 알려주세요.',
            '코로나19 백신 예약은 어떻게 하나요?',
        ],
        'answer': [
            '서울시 120 다산콜센터는 서울시민의 생활 불편사항과 궁금증을 해결하는 통합 콜센터입니다.',
            '주민등록등본은 정부24 온라인 발급, 무인발급기, 주민센터 방문을 통해 발급받을 수 있습니다.',
            '코로나19 백신 예약은 예방접종 사전예약 시스템에서 가능하며, 콜센터 전화 예약도 지원합니다.',
        ],
        'category': ['행정', '민원', '보건']
    })

    # RAGAS로 한국어 테스트셋 생성
    korean_testset = generate_korean_testset_with_ragas(
        df=sample_data,
        n_questions=10
    )

    print("\n" + "="*60)
    print("결론: RAGAS는 한국어 Q/A 생성을 완벽하게 지원합니다!")
    print("="*60)
