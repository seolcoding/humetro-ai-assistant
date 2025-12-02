# Large Language Models: A Survey - 문헌 리뷰

## 1. 논문 정보

- **제목**: Large Language Models: A Survey
- **저자**: Shervin Minaee (Amazon), Tomas Mikolov (CIIRC CTU), Narjes Nikzad (Cologne University), Meysam Chenaghlu (Ultimate.ai), Richard Socher (You.com), Xavier Amatriain (Google), Jianfeng Gao (Microsoft Research)
- **출판연도**: 2024 (arXiv preprint, 2025년 3월 23일 버전)
- **출처**: arXiv:2402.06196v3 [cs.CL]
- **페이지 수**: 44페이지
- **분야**: 자연어처리(NLP), 대규모 언어모델, 인공지능

## 2. 핵심 내용 요약

이 논문은 ChatGPT 출시 이후 급격히 발전한 대규모 언어모델(LLM)에 대한 종합적인 서베이 논문이다. LLM의 역사를 통계적 언어모델, 신경망 언어모델, 사전학습 언어모델, 그리고 현재의 LLM으로 구분하여 발전 과정을 설명한다. GPT, LLaMA, PaLM 등 3대 주요 LLM 패밀리를 중심으로 모델 구조, 학습 방법, 평가 지표, 그리고 실제 응용 방법을 체계적으로 정리한다. 특히 LLM의 emergent abilities(창발적 능력)로 in-context learning, instruction following, multi-step reasoning을 강조하며, RAG(Retrieval-Augmented Generation)를 통한 외부 지식 활용과 도구 사용을 통한 LLM 증강 방법을 상세히 다룬다.

## 3. 주요 기여점

### 3.1 체계적인 분류 체계 제시
- LLM을 크기별(Small/Medium/Large/Very Large), 유형별(Foundation/Instruction/Chat), 공개여부별(Public/Private), 출처별(Original/Tuned)로 분류하는 명확한 카테고리 제시

### 3.2 포괄적인 기술 리뷰
- **모델 구축**: Data Cleaning, Tokenization(BPE, WordPiece, SentencePiece), Positional Encoding(APE, RPE, RoPE), Pre-training, Fine-tuning, Alignment(RLHF, DPO, KTO)
- **모델 사용**: Prompt Engineering 기법(Chain-of-Thought, Tree-of-Thought, Self-Consistency, Reflection, Expert Prompting)
- **모델 증강**: RAG, External Tools, LLM Agents

### 3.3 실증적 벤치마크 분석
- 57개 이상의 데이터셋과 벤치마크에 대한 상세한 분석
- 주요 LLM들의 성능을 Common Sense Reasoning, World Knowledge, Coding, Arithmetic Reasoning, Hallucination 등 다양한 측면에서 비교

### 3.4 미래 연구 방향 제시
- Smaller and More Efficient Models
- Post-attention Architectural Paradigms (State Space Models, Mixture of Experts)
- Multi-modal Models
- Security and Ethical AI

## 4. 방법론

### 4.1 LLM 아키텍처
- **Encoder-Only**: BERT, RoBERTa, ALBERT, DeBERTa (언어 이해 작업에 최적)
- **Decoder-Only**: GPT 시리즈, LLaMA (텍스트 생성 작업에 최적)
- **Encoder-Decoder**: T5, BART (sequence-to-sequence 작업에 최적)

### 4.2 학습 프로세스
1. **데이터 준비**: 필터링, 중복 제거, 전처리
2. **토큰화**: 서브워드 기반 토큰화 (BPE, WordPiece, SentencePiece)
3. **사전학습**: Masked Language Modeling (MLM), Causal Language Modeling (CLM)
4. **파인튜닝**: Instruction Tuning, Supervised Fine-Tuning (SFT)
5. **정렬(Alignment)**: RLHF, DPO, KTO를 통한 인간 선호도 학습

### 4.3 효율성 향상 기법
- **Low-Rank Adaptation (LoRA)**: 학습 파라미터 수를 대폭 감소
- **Quantization**: 모델 가중치의 정밀도를 낮춰 크기와 속도 개선
- **Knowledge Distillation**: 대형 모델로부터 소형 모델 학습
- **Zero Redundancy Optimizer (ZeRO)**: 메모리 효율적 학습

## 5. 실험 결과

### 5.1 모델 성능 비교 (주요 벤치마크)

#### Common Sense Reasoning (HellaSwag)
- GPT-4: **95.3%** (최고 성능)
- Falcon 180B: 88.86%
- LLaMA 2 70B: 87.33%
- Gemini Ultra: 87.8%

#### Arithmetic Reasoning (GSM8K)
- Gemini Ultra: **94.4%**
- GPT-4: 87.1%
- ToRA 70B: 84.3%
- Gemini Pro: 86.5%

#### Coding (HumanEval)
- Gemini Ultra: **74.4%**
- Gemini Pro: 67.7%
- GPT-4: 67.0%
- WizardCoder 15B: 57.3%

#### World Knowledge (TriviaQA)
- PaLM 2-L: **86.1%**
- LLaMA 2 70B: 85.0%
- PaLM-540B: 81.4%

### 5.2 Hallucination 평가
- GPT-4: HHEM 97% (가장 신뢰성 높음)
- GPT-4 Turbo: 97%
- GPT-3.5 Turbo: 96.5%
- LLaMA 2 70B: 94.9%

### 5.3 주요 발견사항
1. **모델 크기와 성능**: 일반적으로 파라미터 수가 많을수록 성능이 우수하지만, 효율적인 학습 기법(instruction tuning, alignment)을 통해 소형 모델도 경쟁력 확보 가능
2. **Emergent Abilities**: 일정 규모 이상(수십억 파라미터)에서 in-context learning, chain-of-thought reasoning 등의 창발적 능력 발현
3. **Domain-specific Performance**: 특정 도메인(코드 생성, 수학 문제 해결)에서는 전문화된 모델이 범용 모델보다 우수

## 6. 우리 연구와의 관련성

### 6.1 On-premise 오픈소스 RAG 시스템 구축
본 연구는 한국어 공공 행정문서를 위한 on-premise RAG 시스템 구축을 목표로 하는데, 이 논문의 다음 내용이 직접적으로 활용 가능:

- **RAG 아키텍처 설계**: Section IV-C의 RAG 구성요소(Retrieval, Generation, Augmentation) 분석
- **오픈소스 LLM 선택**: LLaMA 패밀리, Mistral 등 공개된 모델의 성능 비교 결과 활용
- **한국어 지원**: mT5, BLOOM 등 다국어 모델의 성능 평가 참고

### 6.2 평가 방법론
- **메트릭 선정**: Section VI-A의 평가 메트릭(Exact Match, F1, ROUGE, BLEU, Pass@k) 활용
- **벤치마크 구축**: 행정문서 QA를 위한 자체 벤치마크 설계 시 SQuAD, Natural Questions 등의 구조 참고
- **Hallucination 평가**: 행정 분야의 factual correctness 평가를 위한 방법론 적용

### 6.3 시스템 최적화
- **효율성 개선**: LoRA, Quantization 등의 기법을 통해 제한된 리소스(RTX 3090Ti 24GB)에서도 효과적인 모델 운영 가능
- **프롬프트 엔지니어링**: Chain-of-Thought, Self-Consistency 등을 한국어 행정문서 이해에 적용

### 6.4 인용 포인트
1. LLM의 한계점(hallucination, 최신 정보 부족) → RAG의 필요성 정당화
2. 오픈소스 모델의 성능 향상 추세 → on-premise 시스템의 실현 가능성 입증
3. 평가 메트릭 및 벤치마크 설계 방법론 참조

## 7. 인용 가능한 핵심 문장

### 7.1 LLM의 정의와 특징
> "Large language models mainly refer to transformer-based neural language models that contain tens to hundreds of billions of parameters, which are pre-trained on massive text data... Compared to PLMs, LLMs are not only much larger in model size, but also exhibit stronger language understanding and generation abilities, and more importantly, emergent abilities that are not present in smaller-scale language models."

**한글 번역**: "대규모 언어모델은 주로 수백억 개의 파라미터를 포함하는 트랜스포머 기반 신경망 언어모델을 의미하며, 대규모 텍스트 데이터로 사전학습된다... 사전학습 언어모델(PLM)과 비교하여, LLM은 모델 크기가 훨씬 클 뿐만 아니라 더 강력한 언어 이해 및 생성 능력을 보여주며, 더 중요하게는 소규모 언어모델에서는 나타나지 않는 창발적 능력을 보인다."

### 7.2 RAG의 필요성
> "One of the main limitations of pre-trained LLMs is their lack of up-to-date knowledge or access to private or use-case-specific information. This is where retrieval augmented generation (RAG) comes into the picture. RAG involves extracting a query from the input prompt and using that query to retrieve relevant information from an external knowledge source."

**한글 번역**: "사전학습된 LLM의 주요 한계 중 하나는 최신 지식의 부족이나 비공개 또는 특정 사용 사례에 맞는 정보에 대한 접근성 부재이다. 이러한 상황에서 검색 증강 생성(RAG)이 등장한다. RAG는 입력 프롬프트에서 쿼리를 추출하고 이 쿼리를 사용하여 외부 지식 소스에서 관련 정보를 검색하는 것을 포함한다."

### 7.3 Hallucination 문제
> "Hallucination in an LLM is characterized as 'the generation of content that is nonsensical or unfaithful to the provided source.' LLMs, trained on diverse datasets including the internet, books, and Wikipedia, generate text based on probabilistic models without an inherent understanding of truth or falsity."

**한글 번역**: "LLM의 환각(hallucination)은 '제공된 소스에 대해 무의미하거나 충실하지 않은 콘텐츠의 생성'으로 특징지어진다. 인터넷, 책, 위키피디아를 포함한 다양한 데이터셋으로 학습된 LLM은 진실이나 거짓에 대한 본질적 이해 없이 확률적 모델을 기반으로 텍스트를 생성한다."

### 7.4 오픈소스 모델의 가능성
> "The open-source LLaMA-13B model outperforms the proprietary GPT-3 (175B) model on most benchmarks, making it a good baseline for LLM research."

**한글 번역**: "오픈소스 LLaMA-13B 모델은 대부분의 벤치마크에서 독점 소유인 GPT-3(175B) 모델을 능가하여, LLM 연구를 위한 좋은 기준선이 된다."

### 7.5 Instruction Tuning의 효과
> "Instruction tuned models outperform their original foundation models they are based on. For example, InstructGPT outperforms GPT-3 on most benchmarks. The same is true for Alpaca when compared to LLaMA."

**한글 번역**: "Instruction tuning된 모델은 기반이 되는 원래의 foundation 모델을 능가한다. 예를 들어, InstructGPT는 대부분의 벤치마크에서 GPT-3를 능가한다. Alpaca와 LLaMA를 비교할 때도 마찬가지이다."

### 7.6 효율적인 학습 기법
> "LoRA reduces the number of trainable parameters by learning pairs of rank-decomposition matrices while freezing the original weights. This vastly reduces the storage requirement for large language models adapted to specific tasks and enables efficient task-switching during deployment all without introducing inference latency."

**한글 번역**: "LoRA는 원래 가중치를 고정하면서 순위 분해 행렬 쌍을 학습함으로써 학습 가능한 파라미터 수를 줄인다. 이는 특정 작업에 적응된 대규모 언어모델의 스토리지 요구사항을 크게 줄이고, 추론 지연 시간을 증가시키지 않으면서 배포 중 효율적인 작업 전환을 가능하게 한다."

### 7.7 평가의 중요성
> "Evaluating the performance of LLMs poses particular challenges due to the evolving landscape of their applications. The original intent behind developing LLMs was to boost the performance of NLP tasks... However, it is evident today that these models are finding utility across diverse domains including code generation and finance."

**한글 번역**: "LLM의 성능 평가는 응용 분야의 진화하는 환경으로 인해 특별한 과제를 제기한다. LLM 개발의 원래 의도는 NLP 작업의 성능을 향상시키는 것이었다... 그러나 오늘날 이러한 모델이 코드 생성 및 금융을 포함한 다양한 분야에서 유용성을 찾고 있다는 것이 분명하다."

## 8. 한계점 및 향후 연구방향

### 8.1 논문의 한계점
1. **빠른 발전 속도**: LLM 분야가 매우 빠르게 발전하여 논문 출판 시점에 이미 새로운 모델과 기법이 등장할 수 있음
2. **독점 모델의 제한적 정보**: GPT-4 등 비공개 모델에 대한 상세 정보가 부족하여 완전한 분석이 어려움
3. **한국어 특화 분석 부족**: 주로 영어 중심의 모델과 벤치마크에 초점을 맞춰 한국어 성능에 대한 심층 분석이 제한적

### 8.2 제시된 미래 연구방향

#### 8.2.1 Smaller and More Efficient Models
- Small Language Models (SLMs)의 발전: Phi-1, Phi-1.5, Phi-2
- Parameter-Efficient Fine-Tuning (PEFT) 기법 개선
- Knowledge Distillation을 통한 경량화

#### 8.2.2 Post-attention Architectural Paradigms
- **State Space Models (SSMs)**: S4, Mamba, Hyena 등
- **Mixture of Experts (MoE)**: 학습 시 대규모, 추론 시 효율적
- **Monarch Mixer**: 새로운 sub-quadratic 아키텍처
- **긴 컨텍스트 지원**: 현재 attention 기반 모델의 주요 한계 극복

#### 8.2.3 Multi-modal Models
- LLAVA, GPT-4 Vision, Qwen-VL, Next-GPT 등
- 텍스트, 이미지, 오디오, 비디오의 통합 처리
- 의료, 로봇공학 등 다양한 응용 분야 확장

#### 8.2.4 Improved Usage and Augmentation
- **LLM-based Agents**: 외부 도구 사용 및 의사결정 능력
- **Multi-agent Systems**: 협업적 문제 해결
- **Personalization**: 사용자 맞춤형 상호작용
- **Advanced RAG**: 더욱 정교한 검색 및 생성 메커니즘

#### 8.2.5 Security and Ethical AI
- Adversarial attacks에 대한 강건성
- 편향성(bias) 감소 및 공정성 개선
- 책임 있는 AI 개발
- 잘못된 정보 확산 방지

### 8.3 우리 연구에 적용 가능한 향후 방향
1. **한국어 특화 벤치마크 개발**: 행정문서 도메인에 특화된 평가 데이터셋 구축
2. **효율적 모델 선택**: 7B-13B 규모의 오픈소스 모델(LLaMA 2, Mistral)로 실용적 성능 달성
3. **Hybrid RAG**: Vector-based retrieval + Knowledge Graph 결합
4. **Hallucination 완화**: 행정 분야의 factual correctness를 위한 검증 메커니즘 구축

## 9. 참고문헌 활용 가치

이 논문은 220개 이상의 참고문헌을 포함하여 LLM 분야의 거의 모든 주요 연구를 망라한다. 특히 다음 영역에서 추가 참고문헌 탐색이 가능:

- **RAG 관련**: Lewis et al. (2020), Gao et al. (2023)
- **평가 메트릭**: Lin (2004, ROUGE), Papineni et al. (2002, BLEU)
- **오픈소스 도구**: LangChain, LlamaIndex, HuggingFace 라이브러리
- **한국어 LLM**: 논문에서 다루지 않은 EXAONE, Polyglot-Ko 등은 별도 조사 필요

## 10. 연구 활용 제언

### 10.1 즉시 적용 가능한 내용
- LoRA를 활용한 효율적인 파인튜닝 전략
- RAG 시스템 구축을 위한 3단계 파이프라인(Retrieval-Augmentation-Generation)
- Chain-of-Thought 프롬프팅을 통한 복잡한 행정 문서 이해

### 10.2 심화 연구가 필요한 영역
- 한국어 행정 용어에 대한 토큰화 전략 최적화
- 행정문서 특유의 구조(조문, 표, 부록 등)를 반영한 retrieval 메커니즘
- 공공 분야의 책임 있는 AI 운영을 위한 hallucination 감지 및 완화 기법

---

**작성일**: 2025-11-30
**작성자**: Claude (AI Assistant)
**버전**: 1.0
