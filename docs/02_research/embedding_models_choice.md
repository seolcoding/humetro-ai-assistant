논문의 임베딩 모델 선택 근거 섹션을 작성해드리겠습니다:

---

## 3.X 임베딩 모델 선택

본 연구에서는 한국어 텍스트의 벡터 임베딩을 위해 KURE-v1(Korea University Retrieval Embedding)[^1] 모델을 채택하였다. KURE-v1은 고려대학교 NLP&AI 연구실에서 2024년 12월 공개한 최신 한국어 특화 임베딩 모델로, 한국어 검색 및 정보 검색(Information Retrieval) 작업에 최적화되어 있다.

### 3.X.1 선택 근거

KURE-v1을 선택한 주요 근거는 다음과 같다:

**1) 한국어 검색 특화 설계**

KURE-v1은 BGE-M3(BAAI General Embedding Multilingual-3)[^2] 아키텍처를 기반으로 한국어 triplet 데이터셋을 활용하여 파인튜닝된 모델이다[^1]. 본 연구의 대상인 서울시 다산콜센터 상담 데이터는 한국어 구어체와 행정 용어가 혼재된 특성을 가지고 있어, 한국어 검색에 특화된 임베딩 모델의 필요성이 높았다.

**2) MTEB-ko 벤치마크 성능**

KURE-v1은 MTEB-ko(Massive Text Embedding Benchmark - Korean) 리더보드의 retrieval 태스크에서 우수한 성능을 입증하였다[^3]. 특히 한국어 문서 검색 및 의미적 유사도 측정에서 기존 multilingual 모델 대비 향상된 성능을 보였다.

**3) 오픈소스 접근성**

KURE-v1은 Hugging Face Model Hub를 통해 공개되어 있으며(nlpai-lab/KURE-v1)[^4], Sentence-Transformers 라이브러리와의 호환성이 보장되어 재현 가능한 연구 환경 구축이 용이하다.

**4) 벡터 차원 및 계산 효율성**

KURE-v1은 1024차원의 dense vector를 생성하며[^3], 이는 고차원 검색 정확도와 계산 효율성 간의 균형을 제공한다. 본 연구의 Graph RAG 시스템에서 대량의 청크(chunk) 임베딩을 처리해야 하는 점을 고려할 때, 적절한 차원 수는 중요한 선택 요소였다.

### 3.X.2 구현

KURE-v1 모델은 다음과 같이 구현하였다:

```python
from sentence_transformers import SentenceTransformer

# KURE-v1 모델 로드
embedding_model = SentenceTransformer(
    "nlpai-lab/KURE-v1",
    device="cuda"  # GPU 활용
)

# 텍스트 임베딩 생성
embeddings = embedding_model.encode(
    text_chunks,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True  # 코사인 유사도 계산 최적화
)
```

본 연구에서는 NVIDIA A100 GPU를 활용하여 약 X개의 텍스트 청크를 평균 Y초 내에 임베딩 처리하였다.

---

## References

[^1]: NLP&AI Lab, Korea University. (2024). *KURE: Korea University Retrieval Embedding*. GitHub. <https://github.com/nlpai-lab/KURE>

[^2]: Jang, Y. (2024, December 23). [KURE] 최초의 한국어 특화 임베딩 모델. *Medium*. <https://medium.com/@youngjoon.jang/kure-최초의-한국어-특화-임베딩-모델>

[^3]: 임희석 교려대 교수팀, 한국어 특화 임베딩 모델 'KURE' 공개. (2024). *Newsis*. <https://newsis.com>

[^4]: NLP&AI Lab. (2024). *nlpai-lab/KURE-v1*. Hugging Face Model Hub. <https://huggingface.co/nlpai-lab/KURE-v1>

---

**추가 제안:**

1. **비교 실험 추가**: KURE-v1과 다른 모델(BGE-M3, ko-sroberta-multitask 등)과의 성능 비교 실험 결과를 Table로 추가하면 선택 근거가 더 강화됩니다.

2. **정량적 수치**: 실제 실험에서 측정한 임베딩 속도, 검색 정확도(Recall@k, MRR 등) 수치를 포함하세요.

3. **Citation 형식**: 학교/학회 요구사항에 맞춰 APA, IEEE, 또는 국문 학위논문 형식으로 조정하세요.

논문의 다른 섹션이나 추가 설명이 필요하시면 말씀해주세요!
