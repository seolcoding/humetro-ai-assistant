# RAG 시스템 구축과 평가: 한국 도시철도 지식 어시스턴트 구현하기

## 1. 소개

최근 대규모 언어 모델(LLM)이 급속도로 발전하면서 다양한 분야에 활용되고 있지만, 특정 도메인에 특화된 지식을 제공하거나 최신 정보를 반영하는 데에는 여전히 한계가 있습니다. 이러한 문제를 해결하기 위해 RAG(Retrieval-Augmented Generation) 기술이 주목받고 있습니다. RAG는 외부 데이터 소스에서 관련 정보를 검색하여 LLM의 생성 과정에 통합함으로써 보다 정확하고 신뢰할 수 있는 응답을 생성합니다.

이 블로그 포스트에서는 한국 도시철도(부산교통공사) 역무 지식에 특화된 RAG 시스템을 구축하고 평가한 전체 과정을 다룹니다. 데이터 수집부터 모델 평가까지, 효과적인 RAG 파이프라인을 구축하기 위한 모든 단계를 상세히 설명하겠습니다.

## 2. RAG(Retrieval-Augmented Generation) 이해하기

### 2.1 RAG란 무엇인가?

RAG(Retrieval-Augmented Generation)는 정보 검색(Retrieval)과 텍스트 생성(Generation)을 결합한 방법론입니다. 이 기술은 LLM이 학습 데이터에 포함되지 않은 정보나 최신 정보에 접근할 수 있게 해주며, 응답의 정확성과 신뢰성을 크게 향상시킵니다.

RAG 시스템의 핵심 구성 요소는 다음과 같습니다:

1. **검색기(Retriever)**: 사용자 질의와 관련된 문서나 정보를 검색하는 컴포넌트입니다. 대개 벡터 데이터베이스와 임베딩 모델을 활용합니다.
2. **생성기(Generator)**: 검색된 정보를 바탕으로 응답을 생성하는 LLM 컴포넌트입니다.
3. **프롬프트 엔지니어링**: 검색된 컨텍스트와 사용자 질의를 효과적으로 결합하는 전략입니다.

### 2.2 RAG의 장점

전통적인 LLM 접근법과 비교했을 때 RAG가 제공하는 주요 이점은 다음과 같습니다:

- **최신 정보 활용**: 모델 재학습 없이도 최신 정보에 접근할 수 있습니다.
- **환각(Hallucination) 감소**: 외부 소스에서 검색한 팩트를 기반으로 응답을 생성하므로 잘못된 정보 생성 위험이 감소합니다.
- **검증 가능성**: 응답의 출처를 추적할 수 있어 신뢰성이 향상됩니다.
- **도메인 특화**: 특정 분야의 전문 지식을 효과적으로 활용할 수 있습니다.

## 3. 프로젝트 개요: Humetro AI Assistant

### 3.1 프로젝트 목표

Humetro AI Assistant 프로젝트는 부산 도시철도 역무 지식에 특화된 질의응답 시스템을 개발하는 것을 목표로 합니다. 이 시스템은 다음과 같은 역할을 수행합니다:

- 역무 관련 질문에 정확하고 상세한 답변 제공
- 한국어 도메인에 특화된 정보 검색 및 생성
- 다양한 오픈소스 및 상용 LLM 모델의 성능 비교 평가

### 3.2 도메인 컨텍스트

이 프로젝트는 부산교통공사의 역무 교육 자료를 주요 데이터 소스로 활용합니다. 이러한 도메인 지식은 일반적인 LLM의 학습 데이터에 포함되지 않은 전문적인 내용을 담고 있습니다. 도시철도 운영에 관한 정책, 절차, 규정 등의 정보는 매우 전문적이고 지역 특화적이므로 RAG 접근법이 특히 효과적입니다.

### 3.3 기술 스택

이 프로젝트에서 사용된 주요 기술과 프레임워크는 다음과 같습니다:

- **LangChain**: 문서 로딩, 분할, RAG 파이프라인 구축
- **OpenAI API**: 임베딩 및 일부 언어 모델에 활용
- **Chroma DB**: 벡터 데이터베이스
- **RAGAS**: RAG 시스템 평가 및 합성 데이터셋 생성
- **다양한 LLM 모델**: 로컬에서 실행 가능한 오픈소스 모델과 클라우드 기반 모델

## 4. RAG 파이프라인 구축

### 4.1 데이터 수집 및 전처리

RAG 시스템의 첫 단계는 양질의 데이터를 수집하고 전처리하는 것입니다. 이 프로젝트에서는 다음과 같은 자료를 활용했습니다:

- **야 너두 역무전문가**: 부산교통공사 역무 지식 교육 자료
- **역무지식 100제**: 핵심 역무 지식 문답집
- **일타 역무**: 역무 업무 관련 상세 안내서
- **직원 교육 표준자료**: 공식 역무 교육 자료

이러한 문서들은 원래 한글 워드프로세서(HWP) 형식으로 제공되었으며, 다음과 같은 과정을 통해 전처리되었습니다:

1. **HWP → HTML 변환**: hwp5html 라이브러리를 활용하여 HTML 형식으로 변환
2. **HTML → Markdown 변환**: 구조화된 텍스트 형식으로 추가 변환
3. **메타데이터 추가**: 각 문서에 출처, 제목 등의 메타데이터 추가

```python
def load_documents(doc_dir: str) -> List[Document]:
    """특정 디렉토리에서 마크다운 문서를 로드합니다."""
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
```

### 4.2 문서 분할 및 임베딩

RAG의 효율적인 검색을 위해 문서를 적절한 크기의 청크로 분할했습니다:

```python
def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[Document]:
    """문서를 청크로 분할하면서 메타데이터를 유지합니다."""
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
```

이 프로젝트에서는 800 토큰 크기의 청크와 100 토큰의 오버랩을 사용했습니다. 이는 다음과 같은 이유로 결정되었습니다:

- **적절한 컨텍스트 크기**: 너무 작은 청크는 충분한 컨텍스트를 제공하지 못하고, 너무 큰 청크는 관련 없는 정보를 포함할 수 있습니다.
- **토큰 제한 고려**: LLM의 컨텍스트 창 크기(일반적으로 4K-8K 토큰)를 고려하여 청크 크기를 최적화했습니다.
- **검색 정확도 균형**: 오버랩을 통해 청크 경계에서 정보가 손실되는 것을 방지합니다.

### 4.3 벡터 데이터베이스 구축

다음 단계는 분할된 청크를 벡터 임베딩으로 변환하고 벡터 데이터베이스에 저장하는 것입니다:

```python
def create_embeddings(model_name="text-embedding-3-small"):
    """OpenAI 임베딩 모델을 생성합니다."""
    return OpenAIEmbeddings(model=model_name, dimensions=1536)

def create_vectorstore(splits, embeddings, persist_dir, collection_name="rag_documents"):
    """문서 청크를 임베딩하여 Chroma 벡터스토어에 저장합니다."""
    os.makedirs(persist_dir, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    vectorstore.persist()
    print(f"벡터스토어 생성 완료. 문서 수: {vectorstore._collection.count()}")
    
    return vectorstore
```

이 프로젝트에서는 다음과 같은 선택을 했습니다:

- **임베딩 모델**: OpenAI의 text-embedding-3-small 모델 (1,536 차원)
- **벡터 데이터베이스**: Chroma DB (오픈소스, 로컬 저장소)
- **영속성**: 로컬 디스크에 저장하여 재사용 가능

### 4.4 합성 데이터셋 생성

RAG 시스템의 평가를 위해 RAGAS 프레임워크를 사용하여 합성 QA 데이터셋을 생성했습니다:

```python
async def generate_qa_dataset(
    splits, output_path="datasets/synthetic_qa_dataset.csv", num_questions=200
):
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
    
    return test_df
```

이 과정에서 주목할 점은 다음과 같습니다:

- **질문 유형 분포**: 단일 사실 검색 질문(Single Hop)과 여러 정보를 연결해야 하는 복잡한 질문(Multi Hop)을 7:3 비율로 구성
- **LLM 활용**: GPT-4o와 Grok-3-beta를 사용하여 높은 품질의 질문-답변 쌍 생성
- **한국어 생성**: 도메인 특성에 맞는 한국어 질문-답변 쌍 생성을 위한 프롬프트 조정

## 5. 모델 평가

### 5.1 평가 방법론

RAG 시스템의 성능을 객관적으로 평가하기 위해 RAGAS 프레임워크에서 제공하는 다음 메트릭을 활용했습니다:

1. **Faithfulness (충실도)**: 생성된 응답이 검색된 컨텍스트와 얼마나 일치하는지 측정 (0-1 범위)
2. **Factual Correctness (사실적 정확성)**: 생성된 응답이 레퍼런스 답변과 얼마나 사실적으로 일치하는지 측정 (0-1 범위)

이러한 메트릭은 RAG 시스템의 두 핵심 구성 요소(검색기와 생성기)의 성능을 종합적으로 평가합니다.

### 5.2 평가 대상 모델

다음과 같은 모델들을 평가 대상으로 선정했습니다:

**로컬 실행 모델 (4B 이하 파라미터)**
- HyperCLOVAX-SEED-text-instruct-1.5b-hf-i1
- KakaoCorp.kanana-nano-2.1b-instruct 
- ExaOne-3.5-2.4b-instruct
- Qwen/qwen3-4b:free
- Google/gemma-3-4b-it:free

**대규모 모델**
- Deepseek/deepseek-chat-v3-0324:free (685B, 오픈소스)
- GPT-4o-mini (베이스라인, OpenAI, 8B)

이 모델들은 단일 소비자 GPU(RTX 4090 32GB)에서 실행 가능한 모델들을 중심으로 선정되었습니다.

### 5.3 RAG 체인 구현

모델 평가를 위한 RAG 체인은 다음과 같이 구현되었습니다:

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

### 5.4 평가 결과

각 모델에 대한 RAGAS 평가 결과는 다음과 같습니다:

```python
{
    "hyperclovax_seed_text_1.5b": {
        "faithfulness": 0.3661,
        "factual_correctness(mode=f1)": 0.302,
    },
    "kakaocorp.kanana_nano_2.1b": {
        "faithfulness": 0.4089,
        "factual_correctness(mode=f1)": 0.3714,
    },
    "exaone_3.5_2.4b_instruc": {
        "faithfulness": 0.4023,
        "factual_correctness(mode=f1)": 0.371,
    },
    "google_gemma_3_4b_it_fr": {
        "faithfulness": 0.5984,
        "factual_correctness(mode=f1)": 0.4223,
    },
    "qwen_qwen3_4b_free": {
        "faithfulness": 0.509,
        "factual_correctness(mode=f1)": 0.4147,
    },
    "deepseek_deepseek_chat_": {
        "faithfulness": 0.5682,
        "factual_correctness(mode=f1)": 0.44,
    },
    "gpt_4o_mini": {
        "faithfulness": 0.6388,
        "factual_correctness(mode=f1)": 0.4881,
    }
}
```

이 결과를 시각화하면 다음과 같은 인사이트를 얻을 수 있습니다:

![모델 성능 비교 차트](https://example.com/model_performance_comparison.png)

## 6. 주요 발견점 및 인사이트

### 6.1 모델 크기와 성능 관계

평가 결과에서 확인할 수 있는 가장 명확한 패턴은 모델 파라미터 수와 성능 간의 강한 상관관계입니다:

- 파라미터 수가 클수록 충실도(Faithfulness)와 사실적 정확성(Factual Correctness) 모두에서 더 높은 점수를 기록
- GPT-4o-mini(8B)가 가장 높은 성능을 보였으며, 그 다음으로 Google Gemma-3-4b와 DeepSeek Chat V3가 높은 성능을 보임
- 더 큰 모델일수록 복잡한 지식 기반 질문에 더 정확한 답변을 제공하는 경향이 있음

### 6.2 오픈소스 모델의 가능성

4B 크기의 Gemma3 모델과 같은 오픈소스 모델이 베이스라인 성능(GPT-4o-mini)에 근접하는 성능을 보였다는 점은 매우 고무적입니다:

- 일반 소비자용 그래픽카드 한 장으로도 사용할 수 있는 성능의 RAG 시스템 구축 가능성 확인
- 회사 내부 서버에 GPU 풀링 또는 공공 클라우드 활용으로 공공기관의 전체 수요에 부응하는 private RAG 시스템 구축 가능성 확인

### 6.3 성능 개선 가능성

이 프로젝트에서는 기본적인 Naive RAG 구현만을 사용했으며, 다음과 같은 기법을 적용하여 성능 향상이 가능할 것으로 예상됩니다:

1. **GraphRAG**: 지식 그래프를 활용한 관계 기반 검색 개선
2. **하이브리드 검색**: 키워드 검색과 의미 검색의 조합
3. **문서 최적화**: 청크 크기 및 전처리 방법 개선

## 7. 결론 및 향후 계획

이 프로젝트를 통해 한국 도시철도 역무 지식에 특화된 RAG 시스템의 가능성을 확인하였습니다. 주요 성과는 다음과 같습니다:

1. 도시철도 역무 관련 문서를 체계적으로 수집하고 전처리하는 파이프라인 구축
2. 의미 검색을 위한 벡터 데이터베이스 구축 및 최적화
3. 다양한 오픈소스 및 상용 LLM 모델의 성능 비교 평가
4. 모델 크기와 성능 간의 관계 분석 및 최적 모델 선정

향후 개선 방향으로는 다음과 같은 계획이 있습니다:

1. **Knowledge Graph 구축**: 도시철도 역무 지식을 지식 그래프로 구조화
2. **고급 RAG 기법 적용**: GraphRAG, 하이브리드 검색 등의 고급 기법 적용
3. **모델 평가 및 개선 사이클**: 지속적인 모델 평가 및 튜닝
4. **성능 목표**: 두 메트릭 모두 0.7 이상의 성능을 목표로 파이프라인 개선

이 프로젝트는 특정 도메인 지식을 LLM과 결합하는 효과적인 방법을 보여주며, 앞으로 더 많은 분야에서 RAG 기술이 활용될 수 있는 가능성을 제시합니다.

---

이 블로그 포스트가 여러분의 RAG 시스템 구축에 도움이 되었기를 바랍니다. 질문이나 의견이 있으시면 언제든지 댓글로 남겨주세요!
