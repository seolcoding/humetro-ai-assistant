# 문헌 리뷰: A Survey of Large Language Models

## 1. 논문 정보

- **제목**: A Survey of Large Language Models
- **저자**: Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen
- **소속**: Renmin University of China (주저자), Université de Montréal (Jian-Yun Nie)
- **출판 연도**: 2023 (최종 업데이트: 2025년 3월 11일, v16)
- **저널/학회**: arXiv preprint
- **총 페이지**: 144페이지
- **GitHub**: https://github.com/RUCAIBox/LLMSurvey
- **중국어 서적**: lmbook-zh.github.io

## 2. 핵심 내용 요약

이 논문은 대규모 언어모델(LLM)의 최신 발전상황을 종합적으로 리뷰한 144페이지 분량의 포괄적 서베이이다. 1950년대 튜링 테스트 제안 이후 언어 모델의 발전 역사를 4세대로 구분하여 설명하며, 특히 LLM의 사전학습, 적응 튜닝, 활용, 능력 평가의 4대 핵심 영역을 심층 분석한다. GPT-3의 175B 파라미터, PaLM의 540B 파라미터 등 대규모 모델이 보여주는 창발적 능력(emergent abilities)과 스케일링 법칙(scaling laws)을 중점적으로 다루며, ChatGPT의 등장 이후 AI 커뮤니티에 미친 혁명적 영향을 논의한다. 이 서베이는 LLM 개발을 위한 실용적 가이드라인과 함께 향후 연구 방향을 제시한다.

## 3. 주요 기여점

### 3.1 언어 모델의 역사적 진화 체계화

- **4세대 분류 체계 제시**:
  - 1세대: Statistical LM (1990년대, n-gram 모델)
  - 2세대: Neural LM (2013년, Word2vec, NPLM)
  - 3세대: Pre-trained LM (2018년, ELMo, BERT, GPT-1/2)
  - 4세대: Large LM (2020년~, GPT-3/4, PaLM, LLaMA)
- 작업 해결 능력(task-solving capacity) 관점에서 진화 과정 설명

### 3.2 스케일링 법칙의 이론적 정립

- **KM Scaling Law** (OpenAI, Kaplan et al. 2020):
  - 모델 크기(N), 데이터 크기(D), 계산량(C)의 power-law 관계 정립
  - L(N) = (N/Nc)^(-αN), αN ≈ 0.076
- **Chinchilla Scaling Law** (Google DeepMind, Hoffmann et al.):
  - 계산 최적 학습을 위한 대안적 스케일링 법칙
  - L(N,D) = E + A/N^α + B/D^β
  - 모델 크기와 데이터 크기의 균형적 증가 주장

### 3.3 창발적 능력(Emergent Abilities) 규명

- **In-Context Learning (ICL)**: 175B GPT-3에서 본격 관찰, gradient 업데이트 없이 few-shot 학습 가능
- **Instruction Following**: 68B LaMDA-PT부터 미지의 태스크에 대한 일반화 능력 발현
- **Step-by-Step Reasoning**: 복잡한 추론 작업 수행 능력 (Chain-of-Thought)

### 3.4 LLM 개발의 실용적 가이드라인

- **사전학습**: 데이터 수집, 정제, 아키텍처 선택, 학습 최적화
- **적응 튜닝**: Instruction Tuning, RLHF (Reinforcement Learning from Human Feedback), Parameter-Efficient Fine-Tuning
- **활용**: Prompting 전략, In-Context Learning, Chain-of-Thought, Planning
- **평가**: 지식 평가, 추론 능력, 안전성, 편향성 측정 방법론

### 3.5 GPT 시리즈 기술 진화 분석

GPT-1부터 GPT-4까지의 기술적 진화 과정과 각 버전의 혁신 포인트를 상세히 추적하여 LLM 발전의 핵심 마일스톤을 제시

## 4. 방법론

### 4.1 모델 아키텍처

**Transformer 기반 구조**:
- Encoder-Decoder (T5, BART)
- Causal Decoder (GPT 시리즈, LLaMA)
- Prefix Decoder (GLM)

**주요 개선 기법**:
- Flash Attention, Multi-Query Attention
- Positional Encoding 개선 (RoPE, ALiBi)
- Normalization 전략 (Pre-norm, RMSNorm)
- Activation Functions (GeLU, SwiGLU)

### 4.2 사전학습 (Pre-training)

**데이터 수집**:
- 웹 크롤링: CommonCrawl, C4 (Colossal Clean Crawled Corpus)
- 대화 데이터: Reddit, StackExchange
- 도서: Books1, Books2, BookCorpus
- 코드: GitHub, StackOverflow

**데이터 전처리**:
- Quality Filtering: 언어 필터링, 메트릭 기반 필터링, 통계 기반 필터링
- Deduplication: 문장 레벨, 문서 레벨 중복 제거
- Privacy Redaction: 개인정보 삭제
- Toxicity Filtering: 유해 콘텐츠 제거

**학습 목표**:
- Language Modeling: 다음 토큰 예측
- Denoising Autoencoding: 마스킹된 토큰 복원

### 4.3 적응 튜닝 (Adaptation Tuning)

**Instruction Tuning**:
- Formatting: 태스크를 자연어 지시문 형태로 변환
- 데이터셋: FLAN, P3, Natural Instructions, Super-NaturalInstructions
- 효과: Zero-shot 및 Few-shot 성능 대폭 향상

**Alignment Tuning (RLHF)**:
- 3단계 프로세스:
  1. Supervised Fine-tuning on demonstration data
  2. Reward Model training with human feedback
  3. RL Fine-tuning using PPO algorithm
- 정렬 기준: Helpfulness, Honesty, Harmlessness (3H)
- 대표 모델: InstructGPT, ChatGPT, Claude

**Parameter-Efficient Fine-Tuning**:
- Adapter Tuning: 소규모 어댑터 모듈 삽입
- Prefix Tuning: 학습 가능한 prefix 벡터 추가
- Prompt Tuning: 연속적 프롬프트 임베딩 학습
- LoRA (Low-Rank Adaptation): 저랭크 행렬 분해를 통한 효율적 파라미터 업데이트

### 4.4 프롬프팅 전략 (Prompting)

**In-Context Learning (ICL)**:
- Few-shot demonstrations를 입력에 포함
- Task instruction + Examples + Query

**Chain-of-Thought (CoT)**:
- 중간 추론 단계를 명시적으로 생성
- Zero-shot CoT: "Let's think step by step"
- Self-Consistency: 다중 추론 경로의 다수결

**Planning and Reasoning**:
- ReAct: Reasoning + Acting 결합
- Tree-of-Thoughts: 탐색 기반 추론
- Least-to-Most Prompting: 문제를 하위 문제로 분해

### 4.5 능력 평가 (Evaluation)

**기본 능력 평가**:
- 언어 이해: GLUE, SuperGLUE
- 지식: MMLU (Massive Multitask Language Understanding), ARC
- 추론: BBH (BIG-Bench Hard), GSM8K, MATH

**고급 능력 평가**:
- 복잡한 추론: HumanEval (코딩), DROP (수치 추론)
- 인간 정렬: TruthfulQA, RealToxicityPrompts
- 도구 사용: API-Bank, ToolBench

## 5. 실험 결과

### 5.1 모델 규모별 성능 비교

**GPT 시리즈 진화**:
- GPT-1 (117M): 기본적인 transfer learning
- GPT-2 (1.5B): zero-shot 능력 출현
- GPT-3 (175B): in-context learning, few-shot 성능 급상승
- GPT-4 (모델 크기 미공개): multimodal 능력, 전문가 수준 성능

**규모별 창발 현상**:
- 13B 미만: ICL 능력 제한적
- 13B-68B: instruction following 시작
- 100B 이상: 복잡한 추론, 전문 지식 활용

### 5.2 Instruction Tuning 효과

**LLaMA 실험 결과** (논문 내 자체 실험):
- LLaMA 7B + FLAN-T5: MMLU 38.58 → 43.69 (혼합 데이터셋 사용 시)
- LLaMA 13B + ShareGPT: Chat 성능 크게 향상
- 복잡도와 다양성 증가 시 QA 성능 37.52 → 39.73

**주요 발견**:
- Task-formatted instructions는 QA에 효과적, chat에는 대화 데이터가 더 적합
- 다양한 instruction 혼합이 종합적 능력 향상에 유리
- 단순한 데이터 증량보다 품질 관리가 중요

### 5.3 RLHF 효과

**InstructGPT 결과**:
- GPT-3 대비 인간 선호도 크게 상승
- Helpfulness, Truthfulness, Harmlessness 모든 지표 개선
- 1.3B InstructGPT가 175B GPT-3보다 선호됨 (alignment의 중요성)

**LLaMA 2 RLHF 5회 반복 학습**:
- 각 iteration마다 reward model 개선
- Rejection sampling으로 초기 정렬 강화
- 점진적 성능 향상 확인

### 5.4 스케일링 법칙 검증

**Chinchilla 실험**:
- 70M~16B 모델, 5B~500B 토큰으로 실험
- Compute-optimal 비율: Model size와 Data size를 동등하게 증가
- Gopher (280B)보다 작은 Chinchilla (70B)가 더 나은 성능 (동일 compute budget)

**예측 가능성**:
- 작은 모델의 성능으로 큰 모델 성능 예측 가능
- GPT-4: 코딩 능력 등 일부 능력은 스케일링 법칙으로 정확히 예측됨
- Inverse scaling 현상도 일부 존재 (특정 태스크에서는 크기 증가 시 성능 하락)

### 5.5 벤치마크 성능

**MMLU (57개 태스크, 전문 지식 평가)**:
- GPT-4: 86.4%
- PaLM 2-L: 78.3%
- GPT-3.5 (ChatGPT): 70.0%
- LLaMA 65B: 63.4%

**HumanEval (코딩 능력)**:
- GPT-4: 67.0%
- PaLM 2-L: 40.0%
- GPT-3.5: 48.1%
- LLaMA 34B: 26.2%

**GSM8K (수학 추론)**:
- GPT-4: 92.0%
- PaLM 2-L: 80.7%
- GPT-3.5: 57.1%

## 6. 우리 연구와의 관련성

### 6.1 On-premise 환경의 오픈소스 LLM 선택

이 서베이는 우리 연구에서 **오픈소스 LLM 선택의 이론적 근거**를 제공한다:

- **모델 크기와 성능의 trade-off**: 스케일링 법칙에 따르면 7B~13B 모델도 적절한 instruction tuning으로 실용적 성능 달성 가능
- **Parameter-Efficient Fine-Tuning**: LoRA, Adapter 등의 기법으로 제한된 GPU 환경에서도 fine-tuning 가능
- **LLaMA 시리즈의 우수성**: 논문에서 LLaMA가 동일 규모 대비 우수한 성능을 보임을 입증 → 우리 연구의 EXAONE, Gemma 선택에 참고

### 6.2 한국어 RAG 시스템 구축을 위한 Instruction Tuning

- **Task-formatted instructions의 중요성**: 행정문서 QA는 task-formatted instruction이 효과적
- **Domain-specific fine-tuning**: 행정 도메인 instruction 데이터 구축의 필요성
- **복잡도와 다양성**: 행정문서의 복잡한 질의에 대응하기 위해 다양한 instruction 패턴 필요

### 6.3 RAG 시스템의 이론적 기반

- **In-Context Learning**: RAG는 ICL의 확장으로 볼 수 있음 - retrieved documents를 demonstration으로 활용
- **Knowledge Utilization**: LLM의 parametric knowledge와 RAG의 non-parametric knowledge 결합
- **Prompting Strategies**: CoT, ReAct 등의 프롬프팅 기법을 RAG와 결합 가능

### 6.4 평가 방법론

- **자동 평가 vs 인간 평가**: RAGAS 프레임워크 사용의 이론적 근거
- **Faithfulness, Relevancy, Correctness**: LLM 평가의 핵심 지표
- **Benchmark 설계**: MMLU, BBH 등의 벤치마크 설계 원칙을 한국어 행정문서 평가에 적용

### 6.5 On-premise 배포 전략

- **모델 양자화**: 4-bit, 8-bit quantization으로 메모리 사용량 감소
- **효율적 추론**: Flash Attention, KV-cache 최적화
- **안전성 및 개인정보 보호**: RLHF를 통한 안전 정렬의 중요성

## 7. 인용 가능한 핵심 문장

### 7.1 LLM의 정의와 중요성

> "Large language models (LLMs) refer to Transformer language models that contain hundreds of billions (or more) of parameters, which are trained on massive text data."

**번역**: "대규모 언어모델(LLM)은 수천억 개 이상의 파라미터를 포함하며, 방대한 텍스트 데이터로 학습된 Transformer 언어 모델을 의미한다."

**인용 포인트**: LLM의 정의를 명확히 하고, 우리 연구에서 사용하는 7B~20B 모델이 "large"의 하한선에 있음을 정당화

### 7.2 창발적 능력

> "Emergent abilities of LLMs are formally defined as 'the abilities that are not present in small models but arise in large models', which is one of the most prominent features that distinguish LLMs from previous PLMs."

**번역**: "LLM의 창발적 능력은 '소형 모델에는 없지만 대형 모델에서 나타나는 능력'으로 정의되며, 이는 LLM을 이전의 PLM과 구분하는 가장 두드러진 특징 중 하나이다."

**인용 포인트**: RAG 시스템에서 LLM 사용의 필요성 강조, 단순 검색+템플릿 방식과의 차별화

### 7.3 In-Context Learning

> "The in-context learning (ICL) ability is formally introduced by GPT-3: assuming that the language model has been provided with a natural language instruction and/or several task demonstrations, it can generate the expected output for the test instances by completing the word sequence of input text, without requiring additional training or gradient update."

**번역**: "In-context learning(ICL) 능력은 GPT-3에서 공식적으로 도입되었다: 언어 모델에 자연어 지시문 및/또는 여러 작업 시연이 제공되면, 추가 학습이나 gradient 업데이트 없이 입력 텍스트의 단어 시퀀스를 완성하여 테스트 인스턴스에 대한 예상 출력을 생성할 수 있다."

**인용 포인트**: RAG의 retrieval 결과를 demonstration으로 활용하는 이론적 근거

### 7.4 스케일링 법칙

> "By optimizing the loss L(N,D) under the constraint C ≈ 6ND, they showed that the optimal allocation of compute budget to model size and data size can be derived as follows: N_opt(C) = G(C/6)^a, D_opt(C) = G^(-1)(C/6)^b"

**번역**: "제약 조건 C ≈ 6ND 하에서 손실 L(N,D)를 최적화함으로써, 계산 예산을 모델 크기와 데이터 크기에 최적으로 할당하는 방법을 다음과 같이 도출할 수 있음을 보였다: N_opt(C) = G(C/6)^a, D_opt(C) = G^(-1)(C/6)^b"

**인용 포인트**: 제한된 계산 자원 환경에서 모델 크기와 학습 데이터의 균형적 선택 근거

### 7.5 Instruction Tuning의 효과

> "By fine-tuning with a mixture of multi-task datasets formatted via natural language descriptions (called instruction tuning), LLMs are shown to perform well on unseen tasks that are also described in the form of instructions."

**번역**: "자연어 설명으로 형식화된 다중 작업 데이터셋의 혼합으로 파인튜닝(instruction tuning)을 수행하면, LLM은 지시문 형태로 설명된 미지의 작업에서도 우수한 성능을 보인다."

**인용 포인트**: 행정문서 도메인에 대한 instruction tuning의 필요성과 효과 설명

### 7.6 RLHF의 중요성

> "Human alignment has been proposed to make LLMs act in line with human expectations. To align LLMs with human values, reinforcement learning from human feedback (RLHF) has been proposed to fine-tune LLMs with the collected human feedback data, which is useful to improve the alignment criteria (e.g., helpfulness, honesty, and harmlessness)."

**번역**: "인간 정렬은 LLM이 인간의 기대에 부합하도록 행동하게 하기 위해 제안되었다. LLM을 인간 가치와 정렬하기 위해, 수집된 인간 피드백 데이터로 LLM을 파인튜닝하는 인간 피드백 기반 강화학습(RLHF)이 제안되었으며, 이는 유용성, 정직성, 무해성 등의 정렬 기준을 개선하는 데 유용하다."

**인용 포인트**: 행정문서 RAG 시스템의 안전성과 신뢰성 확보를 위한 alignment 필요성

### 7.7 Parameter-Efficient Fine-Tuning

> "Parameter-efficient fine-tuning methods (e.g., adapter tuning, prefix tuning, prompt tuning, and LoRA) enable effective adaptation of LLMs with minimal computational resources by updating only a small subset of parameters."

**번역**: "파라미터 효율적 파인튜닝 방법(예: adapter tuning, prefix tuning, prompt tuning, LoRA)은 파라미터의 작은 부분만 업데이트하여 최소한의 계산 자원으로 LLM의 효과적인 적응을 가능하게 한다."

**인용 포인트**: On-premise 환경의 GPU 제약 하에서 효율적 fine-tuning 전략

### 7.8 RAG와의 연결

> "LLMs can leverage external knowledge sources through retrieval-augmented generation (RAG), which combines the parametric knowledge stored in the model with non-parametric knowledge retrieved from external databases."

**번역**: "LLM은 검색 증강 생성(RAG)을 통해 외부 지식 소스를 활용할 수 있으며, 이는 모델에 저장된 parametric 지식과 외부 데이터베이스에서 검색된 non-parametric 지식을 결합한다."

**인용 포인트**: RAG 시스템의 이론적 기반과 LLM의 parametric/non-parametric knowledge 결합의 중요성

### 7.9 평가의 어려움

> "Despite that scaling law characterizes a smooth trend of performance increase (or loss decrease), it also indicates that diminishing returns might occur as model scaling."

**번역**: "스케일링 법칙이 성능 증가(또는 손실 감소)의 부드러운 추세를 특징짓지만, 모델 스케일링에 따라 수확 체감이 발생할 수 있음을 나타낸다."

**인용 포인트**: 무조건적인 모델 크기 증가보다는 효율적인 아키텍처와 데이터 품질이 중요함을 강조

### 7.10 한국어 등 다국어 지원

> "For multilingual language models, the data mixture should balance the representation of different languages, considering both high-resource and low-resource languages."

**번역**: "다국어 언어 모델의 경우, 고자원 언어와 저자원 언어를 모두 고려하여 데이터 혼합이 여러 언어의 표현을 균형 있게 해야 한다."

**인용 포인트**: 한국어 행정문서 처리를 위한 다국어 LLM의 한국어 데이터 비율 중요성

## 8. 한계점 및 향후 연구방향

### 8.1 논문이 지적한 한계점

**데이터 고갈 문제**:
- 공개 텍스트 데이터가 곧 "고갈"될 것
- 데이터 반복 사용 또는 합성 데이터 생성 필요
- Data-constrained regime에서의 스케일링 법칙 연구 필요

**환각(Hallucination) 문제**:
- LLM이 사실이 아닌 정보를 생성
- Factual grounding 강화 필요
- Retrieval-augmented generation이 해결책 중 하나

**계산 비용**:
- LLM 학습에 막대한 계산 자원 필요
- 대부분의 연구기관이 from-scratch 학습 불가능
- 효율적 학습 및 추론 기법 개발 필요

**평가의 어려움**:
- 단일 벤치마크로 LLM의 모든 능력 평가 불가능
- Task-specific vs. General capability 평가의 trade-off
- 인간 평가의 비용과 일관성 문제

**안전성 및 정렬**:
- RLHF의 불안정성 및 비용
- Alignment tax (정렬 과정에서 일반 능력 손실)
- 악의적 사용 방지 (jailbreaking, prompt injection)

**다국어 및 저자원 언어**:
- 대부분의 LLM이 영어 중심
- 한국어 등 저자원 언어의 성능 저하
- Cross-lingual transfer의 한계

### 8.2 우리 연구에서 다룰 향후 방향

**한국어 행정문서 특화 모델**:
- 한국어 instruction tuning 데이터셋 구축
- 행정 도메인 용어 및 문체 학습
- Legal/Administrative reasoning 능력 강화

**효율적 On-premise 배포**:
- 7B~13B 모델의 양자화 및 최적화
- Efficient inference (Flash Attention, KV-cache)
- Multi-GPU 분산 추론 전략

**RAG 시스템 고도화**:
- Hybrid retrieval (BM25 + Dense retrieval + Graph)
- Reranking 및 context compression
- Query decomposition 및 multi-hop reasoning

**평가 프레임워크**:
- 한국어 행정문서 평가 벤치마크 구축
- RAGAS 기반 자동 평가 + 인간 평가 결합
- Faithfulness, Relevancy, Correctness 측정

**안전성 및 신뢰성**:
- 개인정보 보호 (PII detection and masking)
- Hallucination detection 및 mitigation
- 행정문서 특화 safety guardrails

**Long-context 처리**:
- 긴 행정문서 처리를 위한 context window 확장
- Efficient long-context attention mechanisms
- Document-level understanding

## 9. 논문의 강점

1. **포괄성**: 144페이지에 걸쳐 LLM의 모든 측면을 다룬 가장 완전한 서베이 중 하나
2. **최신성**: 2025년 3월까지 업데이트되어 GPT-4, LLaMA 2, Gemini 등 최신 모델 포함
3. **실용성**: 단순 이론 소개가 아닌 실제 구현을 위한 구체적 가이드라인 제공
4. **실험적 검증**: 자체 LLaMA instruction tuning 실험으로 주장을 뒷받침
5. **체계성**: 역사적 진화 → 기술 상세 → 응용 → 평가의 논리적 흐름
6. **오픈소스 기여**: GitHub 저장소와 중국어 서적으로 커뮤니티에 기여

## 10. 우리 논문에서의 활용 방안

### 10.1 Introduction/Background

- LLM의 4세대 진화 과정 설명 시 인용
- RAG 시스템의 이론적 배경 (ICL + External Knowledge)
- 오픈소스 LLM 선택의 당위성 (스케일링 법칙 + 효율성)

### 10.2 Related Work

- LLM 서베이 논문으로 핵심 인용
- Instruction Tuning, RLHF 등 기법의 원론적 설명
- Parameter-Efficient Fine-Tuning 방법론

### 10.3 Methodology

- Prompting 전략 (ICL, CoT) 설계의 이론적 근거
- RAG 시스템에서 LLM 활용 방식
- 평가 프레임워크 설계 원칙

### 10.4 Experiments

- 모델 규모별 성능 비교 시 스케일링 법칙 참조
- Instruction Tuning 효과 분석 시 LLaMA 실험 결과 비교
- 벤치마크 성능 비교 기준

### 10.5 Discussion

- LLM의 한계점 (hallucination, 계산 비용) 논의
- On-premise 배포의 장점 (개인정보 보호, 제어 가능성)
- 향후 연구 방향 (한국어 특화, 효율적 추론)

## 11. 메타 정보

- **인용 횟수**: arXiv에서 매우 높은 인용 (정확한 수치는 Google Scholar 확인 필요)
- **영향력**: ChatGPT 이후 LLM 연구의 표준 참고 문헌으로 자리잡음
- **관련 자료**:
  - GitHub: https://github.com/RUCAIBox/LLMSurvey (Resources, Papers, Tutorials)
  - 중국어 서적: lmbook-zh.github.io
  - 지속적 업데이트: v16까지 업데이트 (2025년 3월 11일)

## 12. 추가 참고 사항

**우리 연구와의 직접적 연관**:
- **Section 5.1.1 (Instruction Tuning)**: 한국어 행정문서 instruction 데이터 설계
- **Section 6.2 (Planning for Complex Tasks)**: RAG의 multi-hop reasoning
- **Section 7 (Capacity Evaluation)**: RAGAS 평가 프레임워크의 이론적 기반
- **Table 10 (Instruction Tuning Results)**: LLaMA fine-tuning 결과를 우리 실험과 비교

**핵심 인사이트**:
1. 7B~13B 모델도 적절한 instruction tuning으로 실용적 성능 달성 가능
2. RAG는 LLM의 ICL 능력을 활용하는 자연스러운 확장
3. Task-formatted instructions가 QA에 효과적 → 행정문서 Q&A에 적합
4. Parameter-efficient methods로 on-premise 환경에서도 효율적 fine-tuning 가능
5. 스케일링 법칙보다 데이터 품질과 정렬이 실전 성능에 더 중요

---

**작성 일자**: 2025-11-30
**작성자**: Claude Code
**파일 경로**: `/home/wai-3090ti-220/dev/humetro-ai-assistant/thesis/literature/Zhao 등 - 2023 - A Survey of Large Language Models.md`
