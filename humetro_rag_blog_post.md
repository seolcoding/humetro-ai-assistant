# 한국 도시철도 역무 지식을 위한 RAG 파이프라인 구축과 평가

한국 도시철도 역무 지식에 특화된 RAG(Retrieval-Augmented Generation) 시스템을 개발하고, 오픈소스 및 상용 LLM 모델을 평가해 보았습니다. 본 포스팅에서는 데이터 수집부터 평가까지 전체 과정을 다룹니다.

## 1. 개요

도시철도 역무 분야는 방대하고 전문적인 지식 영역으로, 기존 LLM만으로는 정확한 답변을 제공하기 어렵습니다. 이러한 문제를 해결하기 위해 RAG(Retrieval-Augmented Generation) 기술을 활용하여 다음과 같은 목표를 달성하고자 했습니다:

- 도시철도 역무 관련 문서를 컨텍스트로 LLM에 활용
- 자원 제약이 있는 환경(단일 소비자 GPU)에서 작동 가능한 다양한 로컬 LLM 모델 평가
- 객관적인 평가 방법론을 통한 모델 성능 측정

## 2. 데이터 로드 및 전처리

### 원본 문서

부산교통공사 운영직 교육 자료(공식, 비공식)를 활용했습니다:
- 역무지식 100제
- 야 너두 역무전문가
- 일타 역무
- 직원 교육 표준자료

이러한 문서들은 다음과 같은 과정을 통해 전처리되었습니다:
1. 한글 문서(HWP)를 HTML로 변환 (hwp5html 라이브러리 활용)
2. HTML을 마크다운 형식으로 변환
3. 마크다운 문서를 LangChain의 `DirectoryLoader`와 `UnstructuredMarkdownLoader`를 사용해 로드

```python
def load_documents(doc_dir: str) -> List[Document]:
    """
    특정 디렉토리에서 마크다운 문서를 로드합니다.
    """
    loader = DirectoryLoader(
        doc_dir,
        glob="*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        recursive=False,
    )
    documents = loader.load()
    print(f"로드된 문서 수: {len(documents)}")
    
    # 파일 이름을 metadata에 추가
    for doc in documents:
        if "source" in doc.metadata:
            doc.metadata["filename"] = os.path.basename(doc.metadata["source"])
            
    return documents

# 문서 로드 실행
doc_dir = Path("./datasets/final_docs")
documents = load_documents(doc_dir)  # 총 36개 문서 로드
```

### 문서 분할

RAG 시스템의 효율적인 검색을 위해 문서를 적절한 크기의 청크로 분할했습니다:

```python
def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    문서를 청크로 분할하며 메타데이터를 보존합니다.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"원본 문서 수: {len(documents)}, 분할 후 청크 수: {len(splits)}")
    
    lengths = [len(doc.page_content) for doc in splits]
    print(f"청크 길이 - 평균: {sum(lengths)/len(lengths):.1f}, 최소: {min(lengths)}, 최대: {max(lengths)}")
    
    return splits

# 문서 분할 실행
splits = split_documents(documents)
```

문서 분할 결과 36개의 원본 문서에서 약 480개의 청크가 생성되었으며, 각 청크는 평균 1024자의 길이를 가졌습니다.

## 3. 벡터 데이터베이스 구축

RAG의 검색(Retrieval) 과정에서 의미적 검색(Semantic Search)을 위한 벡터 데이터베이스를 구축했습니다:

```python
def create_embeddings(model_name="text-embedding-3-small"):
    """OpenAI 임베딩 모델을 생성합니다."""
    return OpenAIEmbeddings(model=model_name, dimensions=1536)

def create_vectorstore(splits, embeddings, persist_dir, collection_name="rag_documents"):
    """문서 청크를 임베딩하여 Chroma 벡터스토어에 저장합니다."""
    os.makedirs(persist_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        print(f"기존 벡터스토어 문서 수: {vectorstore._collection.count()}")
        
        if splits:
            vectorstore.add_documents(splits)
            vectorstore.persist()
            print(f"벡터스토어 업데이트 완료. 총 문서 수: {vectorstore._collection.count()}")
    else:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name=collection_name,
        )
        vectorstore.persist()
        print(f"벡터스토어 생성 완료. 문서 수: {vectorstore._collection.count()}")
    
    return vectorstore

# 임베딩 및 벡터스토어 생성
embeddings = create_embeddings()
persist_dir = "vectorstore"
vectorstore = create_vectorstore(splits=splits, embeddings=embeddings, persist_dir=persist_dir)
```

이 과정에서:
- OpenAI의 text-embedding-3-small 모델을 사용하여 문서 임베딩
- 오픈소스 벡터 데이터베이스인 ChromaDB를 사용하여 로컬에 저장
- 효율적인 의미 검색을 위한 인덱스 구축

## 4. 모델 평가를 위한 합성 데이터셋 생성

RAG 시스템의 성능을 객관적으로 평가하기 위해 합성 QA 데이터셋을 생성했습니다:

```python
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer

async def generate_qa_dataset(splits, output_path="datasets/synthetic_qa_dataset.csv", num_questions=200):
    """RAGAS를 사용하여 합성 QA 데이터셋을 생성합니다."""
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
    
    # 질문 유형 분포 설정 (SingleHop 70%, MultiHop 30%)
    distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=llm), 0.7),
        (MultiHopSpecificQuerySynthesizer(llm=llm), 0.3),
    ]
    
    # 한국어로 질문과 답변 생성하도록 프롬프트 변경
    for query, _ in distribution:
        prompts = await query.adapt_prompts(
            "## 매우 중요: **한국어로만 질문과 답변을 생성**, Question and Answer MUST be in KOREAN",
            llm=llm,
        )
        query.set_prompts(**prompts)
    
    # 테스트셋 생성
    print(f"총 {num_questions}개의 QA 쌍 생성 중...")
    testset = generator.generate_with_langchain_docs(
        documents=splits,
        testset_size=num_questions,
        query_distribution=distribution,
    )
    
    # DataFrame으로 변환 및 저장
    test_df = testset.to_pandas()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    test_df.to_csv(output_path, index=False)
    
    print(f"QA 데이터셋 생성 완료: {len(test_df)}개의 질문-답변 쌍")
    print(f"{output_path}에 저장됨")
    
    return test_df

# QA 데이터셋 생성 실행
import asyncio
test_df = asyncio.run(generate_qa_dataset(splits, num_questions=100))
```

이 과정에서:
- X.AI의 Grok-3-beta 및 OpenAI의 GPT-4o를 데이터 생성의 기반 LLM으로 활용
- RAG 시스템 평가에 널리 사용되는 RAGAS 프레임워크 활용
- 싱글 홉 질문(간단한 단일 사실 질문)과 멀티 홉 질문(여러 정보를 연결해야 하는 복잡한 질문)을 7:3 비율로 약 500여개 데이터셋 생성

## 5. RAG 체인 구축 및 모델 평가

### 평가 대상 모델

시스템의 하드웨어 제약을 고려하여 다음 모델들을 평가 대상으로 선정했습니다:

**로컬 실행 모델 (4B 이하)**
- HyperCLOVAX-SEED-text-instruct-1.5b-hf-i1
- KakaoCorp.kanana-nano-2.1b-instruct 
- ExaOne-3.5-2.4b-instruct
- Qwen/qwen3-4b:free
- Google/gemma-3-4b-it:free

**더 큰 모델**
- Deepseek/deepseek-chat-v3-0324:free (685B, 오픈소스)
- GPT-4o-mini (베이스라인, OpenAI, 8B)

### RAG 체인 구현

```python
def create_rag_chain(llm, retriever):
    """RAG 체인을 생성합니다."""
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
```

### 답변 생성 및 평가 과정

1. 합성 데이터셋의 질문을 retriever에 넣어 관련 컨텍스트를 검색
2. 질문과 검색된 컨텍스트를 모델에 제공하여 답변 생성
3. 생성된 답변을 RAGAS 프레임워크로 평가
4. GPT-4o-mini 모델을 베이스라인으로 활용하여 다른 모델들과 비교

## 6. 평가 결과 및 분석

### RAGAS의 기본 평가 메트릭

평가에 사용된 주요 메트릭은 다음과 같습니다:

1. **Faithfulness (충실도)**: 응답이 검색된 컨텍스트와 얼마나 일치하는지 측정
2. **Factual Correctness (사실적 정확성)**: 응답이 레퍼런스 답변과 얼마나 사실적으로 일치하는지 측정

### 평가 결과

```
{
  "hyperclovax_seed_text_1.5b": {
    "faithfulness": 0.3661,
    "factual_correctness(mode=f1)": 0.302
  },
  "kakaocorp.kanana_nano_2.1b": {
    "faithfulness": 0.4089,
    "factual_correctness(mode=f1)": 0.3714
  },
  "exaone_3.5_2.4b_instruc": {
    "faithfulness": 0.4023,
    "factual_correctness(mode=f1)": 0.371
  },
  "google_gemma_3_4b_it_fr": {
    "faithfulness": 0.5984,
    "factual_correctness(mode=f1)": 0.4223
  },
  "qwen_qwen3_4b_free": {
    "faithfulness": 0.509,
    "factual_correctness(mode=f1)": 0.4147
  },
  "deepseek_deepseek_chat_": {
    "faithfulness": 0.5682,
    "factual_correctness(mode=f1)": 0.44
  },
  "gpt_4o_mini": {
    "faithfulness": 0.6388,
    "factual_correctness(mode=f1)": 0.4881
  }
}
```

### 결과 시각화 및 분석

![RAG 성능 비교](rag_performance_comparison.png)

평가 결과에서 다음과 같은 흥미로운 점들을 발견했습니다:

1. **파라미터 수와 성능**: 모델 크기가 클수록 전반적인 성능이 향상됨
   - GPT-4o-mini(8B)가 가장 높은 성능을 보임
   - 그 다음으로 오픈소스 모델인 Google Gemma-3-4b와 DeepSeek Chat V3의 성능이 좋음

2. **충실도 vs 사실 정확성**: 모든 모델에서 충실도(Faithfulness) 점수가 사실 정확성(Factual Correctness) 점수보다 높게 나타남
   - 이는 모델들이 주어진 컨텍스트에 충실하게 답변하지만, "정답"과의 일치도는 다소 낮을 수 있음을 시사

3. **파라미터 효율성**: 작은 모델들의 파라미터당 성능 효율이 더 높음
   - 특히 HyperCLOVAX와 같은 작은 모델들은 파라미터 수 대비 매우 효율적인 성능을 보임

4. **벤치마크 기준**: 업계 기준으로 두 메트릭 모두에서 0.7 이상의 점수를 요구하는 경우가 많으나, 테스트한 모든 모델이 이 기준에 미치지 못함
   - 기본 목표 점수인 0.5에 가장 가까운 모델은 GPT-4o-mini와 Google Gemma-3-4b

## 7. 결론 및 향후 방향

### 주요 발견점

1. **파라미터 수가 클수록 성능이 일관되게 상승**
   - 더 큰 모델일수록 복잡한 지식 기반 질문에 더 정확한 답변을 제공

2. **오픈소스 모델의 가능성 확인**
   - 4B 크기의 Gemma3 모델의 점수는 베이스라인 성능에 근접함
   - 일반 소비자용 그래픽카드 한 장으로도 사용할 수 있는 성능의 RAG 시스템 구축 가능성 확인

3. **Naive RAG의 한계**
   - 기본적인 RAG 구현만으로는 성능 지표가 기대 수준(0.7)에 미치지 못함
   - 향상된 RAG 기법의 필요성 확인

### 향후 개선 방향

현재 구현은 기본적인 Naive RAG만을 사용했으므로, 다음과 같은 기법을 적용하여 성능 향상이 가능할 것으로 예상됩니다:

- **향상된 RAG 기법 적용**
  - GraphRAG: 지식 그래프를 활용한 관계 기반 검색
  - 하이브리드 검색: 키워드 검색과 의미 검색의 조합
  - 문서 최적화: 청크 크기 및 전처리 방법 개선

- **Knowledge Graph와의 통합**
  - 도시철도 역무 지식을 Knowledge Graph로 구조화
  - KG가 적용된 RAG 시스템 구축
  
- **모델 평가 및 개선 사이클**
  - 지속적인 모델 평가 및 튜닝
  - 자연어처리, 트랜스포머, KG, RAG, LLM as a judge 등의 영역에 대한 지속적 학습
  - 두 메트릭 모두 0.7 이상의 성능을 목표로 파이프라인 개선

이 프로젝트를 통해 한국 도시철도 역무 지식에 특화된 RAG 시스템의 가능성을 확인하였습니다. 향후 추가 연구와 개선을 통해 실제 현장에서 활용 가능한 수준의 시스템으로 발전시킬 계획입니다.
