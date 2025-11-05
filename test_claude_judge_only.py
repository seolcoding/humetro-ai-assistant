#!/usr/bin/env python3
"""
Claude Judge 단독 테스트
OpenAI quota 문제를 우회하여 Claude가 RAGAS judge로 작동하는지만 검증
"""

import os
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
from datasets import Dataset

# Claude endpoint 설정
os.environ["CLAUDE_CODE_API_BASE"] = "http://localhost:25292/v1"
os.environ["CLAUDE_CODE_API_KEY"] = "dummy"

print("=" * 70)
print("Claude Judge 단독 테스트")
print("=" * 70)
print()

# 1. Claude LLM 초기화
print("🔧 Claude LLM 초기화 중...")
claude_llm = ChatOpenAI(
    model="claude-sonnet-4-5-20250929",
    openai_api_base="http://localhost:25292/v1",
    openai_api_key="dummy",
    temperature=1.0,
    max_tokens=8192
)

# LangChain wrapper로 감싸기
wrapped_claude = LangchainLLMWrapper(claude_llm)
print("✅ Claude LLM 초기화 완료")
print()

# 2. 테스트용 더미 데이터 생성
print("📝 테스트 데이터 생성 중...")
test_data = {
    "question": ["서울역에서 광화문까지 가는 버스는 무엇인가요?"],
    "answer": ["서울역에서 광화문까지는 160번, 260번, 600번 버스를 이용할 수 있습니다."],
    "contexts": [["서울역에서 광화문까지 가는 버스는 160번, 260번, 600번이 있습니다. 소요시간은 약 15분입니다."]],
    "ground_truth": ["160번, 260번, 600번 버스를 이용할 수 있습니다."]
}

dataset = Dataset.from_dict(test_data)
print("✅ 테스트 데이터 생성 완료")
print(f"   질문: {test_data['question'][0]}")
print(f"   답변: {test_data['answer'][0]}")
print()

# 3. RAGAS metrics 설정 (Claude judge 사용)
print("⚙️  RAGAS metrics 설정 중 (Claude judge)...")
try:
    # Metrics are already instantiated, just need to set LLM
    faithfulness.llm = wrapped_claude
    answer_relevancy.llm = wrapped_claude
    answer_correctness.llm = wrapped_claude

    print("✅ RAGAS metrics 설정 완료")
    print()

except Exception as e:
    print(f"❌ Metrics 설정 실패: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 4. Claude judge로 평가 실행
print("🧪 Claude Judge 평가 시작...")
print()

try:
    from ragas import evaluate

    print("   평가 중...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, answer_correctness],
        llm=wrapped_claude
    )

    print()
    print("=" * 70)
    print("🎉 Claude Judge 테스트 성공!")
    print("=" * 70)
    print()
    print("📊 최종 결과:")
    print(f"   결과 타입: {type(result)}")
    print(f"   결과 키: {result.keys() if hasattr(result, 'keys') else 'N/A'}")
    print(f"   전체 결과: {result}")
    print()
    print("✅ Claude가 RAGAS judge로 정상 작동합니다!")

except Exception as e:
    print(f"❌ 평가 실패: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
