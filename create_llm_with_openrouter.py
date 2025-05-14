from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("HUMETRO_EVAL")


lms_models = [
    "hyperclovax-seed-text-instruct-1.5b-hf-i1",
    "kakaocorp.kanana-nano-2.1b-instruct",
    "exaone-3.5-2.4b-instruct",
    "qwen/qwen3-4b:free",
    "google/gemma-3-4b-it:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "gpt-4o-mini",
]


def create_llm(model_signature: str, temperature: float = 0.1) -> Any:
    OPENROUTER_API_KEY = (
        "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"
    )
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    local_models = [
        "exaone-3.5-2.4b-instruct",
        "kakaocorp.kanana-nano-2.1b-instruct",
        "hyperclovax-seed-text-instruct-1.5b-hf-i1",
    ]
    open_router_models = [
        "qwen/qwen3-4b:free",
        "google/gemma-3-4b-it:free",
        "deepseek/deepseek-chat-v3-0324:free",
    ]
    if "gpt" in model_signature:
        return ChatOpenAI(model=model_signature, temperature=temperature)
    if model_signature in local_models:
        try:
            subprocess.run(["lms", "unload", "-a"])
            print("모든 lms 언로드 완료")
            subprocess.run(["lms", "load", model_signature])
            print(f"lms 모델 {model_signature} 로드 완료")
            return ChatOpenAI(
                base_url="http://localhost:1234/v1", model=model_signature
            )
        except Exception as e:
            print(f"lms 명령 실행 중 오류 발생: {e}")
            print("lms가 설치되어 있고 실행 중인지 확인하세요.")
    if model_signature in open_router_models:
        llm = ChatOpenAI(
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
            model_name=model_signature,
            # model_kwargs={
            #     "headers": {
            #         "HTTP-Referer": getenv("YOUR_SITE_URL"),
            #         "X-Title": getenv("YOUR_SITE_NAME"),
            #     }
            # },
        )
        return llm
    return None
