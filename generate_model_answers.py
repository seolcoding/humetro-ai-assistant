import json
import re
import subprocess
from typing import Any

import pandas as pd
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tqdm import tqdm


# 2. 임베딩 모델 생성
def create_embeddings(model_name: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model_name, dimensions=1536)


def load_vectorstore(persist_directory: str, embeddings: Any) -> Chroma:
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="rag_documents",
    )
    print(f"벡터스토어 로드 완료. 문서 수: {vectorstore._collection.count()}")
    return vectorstore


# 3. 검색기(retriever) 생성 함수
def create_retriever(vectorstore: Chroma, k: int = 4) -> Any:
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


# 4. LLM 모델 생성 함수
def create_llm_ollama(model_name: str, temperature: float = 0.1) -> Any:
    if "ollama" in model_name:
        # Ollama 모델 확인 및 없으면 다운로드
        try:
            result = subprocess.run(
                ["ollama", "ps"], capture_output=True, text=True, check=True
            )
            running_servers = result.stdout.strip()

            # 2. 실행 중인 모델이 있으면 종료
            if running_servers:
                print("현재 실행 중인 모델을 종료합니다...")
                for line in running_servers.split("\n"):
                    if line and not line.startswith("NAME"):  # 헤더 행 제외
                        running_model = line.split()[0]
                        # 실행 중인 모델이 요청된 모델과 다른 경우에만 종료
                        if running_model != model_name:
                            print(f"모델 {running_model}을 종료합니다...")
                            subprocess.run(
                                ["ollama", "stop", running_model],
                                check=True,
                                capture_output=True,
                            )
            # 사용 가능한 모델 목록 확인
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, check=True
            )
            available_models = result.stdout.lower()

            # 모델 이름에서 태그 분리
            if ":" in model_name:
                base_model = model_name.split(":")[0]
            else:
                base_model = model_name

            # 모델이 없으면 다운로드
            if base_model not in available_models:
                print(f"모델 {model_name}을 다운로드합니다...")
                subprocess.run(["ollama", "pull", model_name], check=True)
                print(f"모델 {model_name} 다운로드 완료")
            else:
                print(f"모델 {model_name}이 이미 존재합니다.")

        except subprocess.CalledProcessError as e:
            print(f"Ollama 명령 실행 중 오류 발생: {e}")
            print("Ollama가 설치되어 있고 실행 중인지 확인하세요.")
        except Exception as e:
            print(f"Ollama 모델 준비 중 오류 발생: {e}")

        # Ollama 모델 연결
        return Ollama(model=model_name, temperature=temperature)
    else:
        # OpenAI 모델 사용
        return ChatOpenAI(model=model_name, temperature=temperature)


def create_llm_lms(model_signature: str, temperature: float = 0.1) -> Any:
    if "gpt" in model_signature:
        return ChatOpenAI(model=model_signature, temperature=temperature)
    try:
        subprocess.run(["lms", "unload", "-a"])
        print("모든 lms 언로드 완료")
        subprocess.run(["lms", "load", model_signature])
        print(f"lms 모델 {model_signature} 로드 완료")
    except Exception as e:
        print(f"lms 명령 실행 중 오류 발생: {e}")
        print("lms가 설치되어 있고 실행 중인지 확인하세요.")
    return ChatOpenAI(base_url="http://localhost:1234/v1", model=model_signature)


# 5. RAG 체인 생성 함수
def create_rag_chain(llm: Any, retriever: Any) -> Any:
    # 한국의 역무환경을 고려한 RAG 프롬프트 템플릿
    template = """
당신은 한국의 도시철도 역무 지식 도우미입니다.
주어진 질문에 대해 제공된 문맥 정보를 기반으로 정확하고 도움이 되는 답변을 제공하세요.
문맥에 없는 내용은 답변하지 마세요. 모르는 경우 솔직히 모른다고 말하세요.

문맥 정보:
{context}

질문: {question}

답변:
"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # LCEL을 사용한 RAG 체인 정의
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def sanitize_filename(filename):
    return re.sub(r"[^a-zA-Z0-9_.]", "_", filename)[:20] + ".json"


if __name__ == "__main__":
    lms_models = {
        # "qwen3-1.7b": "qwen3-1.7b",
        "exaone-3.5-2.4b-instruct": "exaone-3.5-2.4b-instruct",
        "kakaocorp.kanana-nano-2.1b-instruct": "kakaocorp.kanana-nano-2.1b-instruct",
        "hyperclovax-seed-text-instruct-1.5b-hf-i1": "hyperclovax-seed-text-instruct-1.5b-hf-i1",
        "qwen3-4b": "qwen3-4b",
        "gpt-4o-mini": "gpt-4o-mini",
    }
    embeddings = create_embeddings()
    vectorstore = load_vectorstore(
        persist_directory="vectorstore", embeddings=embeddings
    )
    if len(vectorstore.similarity_search("서울역 주변 명소")) == 0:
        raise ValueError("vectorstore is empty")
    if len(vectorstore.get()["ids"]) == 0:
        raise ValueError("vectorstore is empty")

    # 질문 데이터 불러오기
    question_data = pd.read_csv("./translated_output.csv")
    questions = list(question_data["user_input"])
    print(questions[:10])

    for model_name, model_signature in lms_models.items():
        try:
            print(f"Loading {model_name}...")
            llm = create_llm_lms(model_signature)
            retriever = create_retriever(vectorstore)
            rag_chain = create_rag_chain(llm, retriever)
            filename = sanitize_filename(f"result_{model_name}.json")
            try:
                with open(filename, "r") as f:
                    result_list = json.load(f)
            except FileNotFoundError:
                result_list = []

            # 이미 처리된 질문 제외
            processed_question = [i["question"] for i in result_list]
            questions = [i for i in questions if i not in processed_question]

            for question in tqdm(questions, desc=f"Evaluating {model_name}"):
                result = rag_chain.invoke(question)
                result_list.append({"question": question, "answer": result})
                if len(result_list) % 10 == 0:  # 10개마다 저장하기
                    print(f"checkpoint, saved {len(result_list)}")
                    with open(filename, "w") as f:
                        json.dump(result_list, f, ensure_ascii=False)
            with open(filename, "w") as f:  # 최종 결과 저장하기
                json.dump(result_list, f, ensure_ascii=False)
        except Exception as e:
            print(f"Error: while generating {model_name}: {e}")
            continue
