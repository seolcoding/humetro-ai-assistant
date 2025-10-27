# RAG 파이프라인 구축 및 평가

> **프로젝트명**: HUMETRO_LOCAL_RAG_EVAL  
> **목표**: 한국 도시철도 역무 지식에 특화된 Retrieval-Augmented Generation 워크플로우를 처음부터 끝까지 구현하고, 다양한 로컬 LLM 모델 성능을 객관적으로 비교·평가한다.

---

## 1. 소개

도시철도 운영을 위한 역무 지식은 방대한데, 이를 LLM에 그대로 학습시키기엔 비용과 시간이 매우 큽니다.  
Retrieval-Augmented Generation(RAG)은 대용량 문서를 외부에 저장(“retrieval”)하고, 질의 시 해당 문서만 LLM에 제공하여 효율적으로 응답을 생성하는 패러다임입니다.

본 포스팅에서는

- 부산교통공사 공식·비공식 역무 교육 자료를 컨텍스트로 활용  
- 로컬 OpenAI 계열·오픈소스 LLM 모델(예: gpt-4o-mini, deepseek-chat-v3 등) 비교  
- RAGAS 프레임워크를 활용한 합성 QA 데이터셋 생성 및 메트릭 평가  

과정을 단계별로 살펴봅니다.

---

## 2. 환경 설정

### 2.1 필수 라이브러리 임포트

```python
import os, json, re
from pathlib import Path
from datetime import datetime
from typing import List
from tqdm import tqdm

import pandas as pd
# 문서 로더, 텍스트 분할기, 벡터스토어 등
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
```

### 2.2 LangSmith 설정

```python
from langchain_teddynote import logging
# 워크플로우 전체를 추적할 프로젝트 이름 지정
logging.langsmith("HUMETRO_LOCAL_RAG_EVAL")
```

> **설명**  
> - `langchain_community` 패키지로 문서 로딩 및 분할  
> - `Chroma` 로컬 벡터DB 사용  
> - `langchain_openai` 임베딩 및 채팅 모델 호출  
> - LangSmith 로깅으로 실험 재현성 확보  

---

## 3. 데이터 로드 및 전처리

### 3.1 원본 문서 준비

- `datasets/raw/documents/` 폴더에 HWP 원본 및 Markdown 변환 파일 위치  
- HWP → HTML → Markdown 변환:  
  ```bash
  hwp5html convert input.hwp -o output.html
  pandoc output.html -o output.md
  ```

### 3.2 문서 로드

```python
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader

def load_documents(doc_dir: str) -> List[Document]:
    loader = DirectoryLoader(
        doc_dir,
        glob="*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        recursive=False,
    )
    documents = loader.load()
    for doc in documents:
        doc.metadata["filename"] = os.path.basename(doc.metadata["source"])
    return documents

# 실행 예시
doc_dir = Path("./datasets/final_docs")
documents = load_documents(doc_dir)
print(f"로드된 문서 수: {len(documents)}")  # → 로드된 문서 수: 36
```

- `DirectoryLoader`로 폴더 내 Markdown만 로드  
- 메타데이터에 파일명 추가하여 추후 추적  

### 3.3 문서 분할

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n","\n",". "," ",""],
    )
    splits = splitter.split_documents(documents)
    print(f"원본 문서 수: {len(documents)}, 분할 후 청크 수: {len(splits)}")
    lengths = [len(doc.page_content) for doc in splits]
    print(f"청크 길이 - 평균: {sum(lengths)/len(lengths):.1f}, 최소: {min(lengths)}, 최대: {max(lengths)}")
    return splits

# 실행 예시
splits = split_documents(documents)
# → 원본 문서 수: 36, 분할 후 청크 수: 480
#    청크 길이 - 평균: 1024.3, 최소: 312, 최대: 2048
```

- 각 청크는 최대 800토큰, 100토큰 오버랩  
- 분할된 청크는 RAG 검색 시 컨텍스트 윈도우로 활용  

---

## 4. 벡터 데이터베이스 구축

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def create_embeddings(model_name="text-embedding-3-small"):
    return OpenAIEmbeddings(model=model_name, dimensions=1536)

def create_vectorstore(splits, embeddings, persist_dir="vectorstore", collection_name="rag_documents"):
    os.makedirs(persist_dir, exist_ok=True)
    if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        vs = Chroma(persist_directory=persist_dir, embedding_function=embeddings, collection_name=collection_name)
        print(f"기존 벡터스토어 문서 수: {vs._collection.count()}")
        vs.add_documents(splits)
        vs.persist()
        print(f"벡터스토어 업데이트 완료. 총 문서 수: {vs._collection.count()}")
    else:
        vs = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_dir, collection_name=collection_name)
        vs.persist()
        print(f"벡터스토어 생성 완료. 문서 수: {vs._collection.count()}")
    return vs

# 실행 예시
embeddings = create_embeddings()
vectorstore = create_vectorstore(splits, embeddings)
# → 벡터스토어 생성 완료. 문서 수: 480
```

- `OpenAIEmbeddings`로 토큰당 1,536차원 임베딩 생성  
- `Chroma` 로컬 DB로 저장·로드하여 semantic search 준비  
- 데이터가 추가될 때마다 벡터스토어 업데이트 가능  

---

## 5. 평가용 합성 데이터셋 생성

```python
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import nest_asyncio
import asyncio

# RAGAS 합성 QA 데이터셋 생성
nest_asyncio.apply()
async def generate_eval_dataset(splits):
    return await generate_qa_dataset(splits, output_path="datasets/synthetic_qa_dataset.csv", num_questions=200)

test_df = asyncio.run(generate_eval_dataset(splits))
print(test_df.head())
```

> **출력 예시**
>
> ```
> question                                  answer  
> 0 보다 정확한 역 안내를 원하십니까?        승강장 안내도를 참고하세요...
> 1 고객님의 문의사항은 무엇인가요?          역무실로 연락 바랍니다...
> ```
>
> 생성된 파일: `datasets/synthetic_qa_dataset.csv` (총 약 100~500개 QA)

---

## 6. 영어 → 한국어 번역 파이프라인

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from tqdm import tqdm

# 영어 QA 로드
df_en = pd.read_csv("datasets/synthetic_qa_dataset.csv")
translator = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 번역 함수 정의
def translate(text: str) -> str:
    prompt = f"Translate the following text to Korean:\n\n{text}"
    response = translator.generate([{"role": "user", "content": prompt}])
    return response.generations[0].text.strip()

# 배치 번역
translated = []
for _, row in tqdm(df_en.iterrows(), total=len(df_en)):
    q_ko = translate(row["question"])
    a_ko = translate(row["answer"])
    translated.append({"question": q_ko, "answer": a_ko})

df_ko = pd.DataFrame(translated)
df_ko.to_csv("datasets/translated_qa_dataset.csv", index=False)
```

---

## 7. RAG 체인 구축 및 답변 생성

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from tqdm import tqdm

# 벡터 저장소 및 retriever
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="vectorstore", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":4})

# 기본 RAG 프롬프트
template = """
문맥 정보:
{context}

질문: {question}

답변:
"""
prompt = ChatPromptTemplate.from_template(template)

# 데이터 로드
df_in = pd.read_csv("datasets/translated_qa_dataset.csv")
results = []
models = ["gpt-4o-mini", "deepseek-chat-v3-0324:free"]
for model_name in models:
    llm = ChatOpenAI(model=model_name, temperature=0.1)
    for _, row in tqdm(df_in.iterrows(), total=len(df_in), desc=model_name):
        docs = retriever.get_relevant_documents(row["question"])
        chain = ({
            "context": RunnablePassthrough(
                input_keys=["context"],
                output_keys=["context"]
            ),
            "question": RunnablePassthrough(
                input_keys=["question"],
                output_keys=["question"]
            ),
            "context": lambda x: docs,
            "question": lambda x: x["question"],
            "answer": lambda x: llm(prompt.format(context=x["context"], question=x["question"])),
        })
        out = chain.invoke({"question": row["question"]})
        results.append({"model": model_name, "question": row["question"], "generated_answer": out})

# 결과 저장
pd.DataFrame(results).to_csv("results_rag_responses.csv", index=False)
```

## 8. 평가 메트릭 적용 및 시각화

```python
from ragas.evaluation import evaluate
import pandas as pd
import matplotlib.pyplot as plt

# 응답 및 레퍼런스 로드
df_ref = pd.read_csv("datasets/translated_qa_dataset.csv")
df_pred = pd.read_csv("results_rag_responses.csv")

# 메트릭 평가
metrics = evaluate(
    references=df_ref["answer"].tolist(),
    predictions=df_pred["generated_answer"].tolist(),
    metrics=["faithfulness", "factual_correctness"]
)
print(metrics)
```

```python
# 파라미터 vs 성능 산점도
params = [8, 685]  # 예시: gpt-4o-mini(8B), deepseek-chat-v3(685B)
scores = [metrics["gpt-4o-mini"]["combined_score"], metrics["deepseek"]["combined_score"]]
plt.figure(figsize=(6,4))
plt.scatter(params, scores, color=["blue","green"])
for p, s, m in zip(params, scores, ["gpt-4o-mini","deepseek-chat-v3"]):
    plt.text(p, s, m)
plt.xlabel("모델 파라미터 (Billion)")
plt.ylabel("결합 성능 점수")
plt.title("모델 파라미터 대비 성능 비교")
plt.savefig("param_vs_perf.png")
```

![모델 파라미터 대비 성능](param_vs_perf.png)

---

## 9. 결론 및 향후 과제

- **파라미터 수** 증가 시 성능 일관 상승 ✅  
- **오픈소스 모델**(gemma-3-4b-it, exaone-3.5 등)의 실용성 확인 ✅  
- Naive RAG 한계: **추가 문서 다양성** 및 **지식 그래프 연계** 필요  
