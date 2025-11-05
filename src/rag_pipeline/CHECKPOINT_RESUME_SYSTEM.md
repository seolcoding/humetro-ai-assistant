# RAG Benchmark Checkpoint/Resume System

## 📋 Overview

통합 벤치마크 시스템에 실험 ID와 체크포인트/재개 기능이 구현되었습니다.

## 🔑 Key Features

### 1. **Experiment ID System**
- 각 실험마다 고유한 정수 ID 자동 부여 (incremental)
- 메타데이터와 체크포인트 파일로 상태 영구 저장
- 실험 구성, 상태, 완료된 단계 추적

### 2. **Checkpoint Mechanism**
- 각 주요 단계 완료시 자동 체크포인트 저장
- Pickle 형식으로 데이터 직렬화
- 단계별 독립적인 체크포인트 파일

### 3. **Resume Capability**
- `--resume-id` 플래그로 중단된 실험 재개
- 마지막 완료 단계 이후부터 자동 재시작
- 체크포인트 데이터 자동 로드

## 📂 Directory Structure

```
data/evaluation/experiments/
├── metadata.json                 # 전체 실험 메타데이터
└── checkpoints/
    ├── exp_1_questions.pkl       # 실험 1 질문 체크포인트
    ├── exp_1_classification.pkl  # 실험 1 분류 체크포인트
    ├── exp_1_model_gpt-4o-mini.pkl
    └── ...
```

## 🗄️ Metadata Structure

```json
{
  "next_id": 4,  // 다음 실험 ID
  "experiments": {
    "1": {
      "id": 1,
      "created_at": "2025-11-05T13:30:00",
      "config": {
        "questions": 50,
        "models": ["gpt-4o-mini", "ollama/exaone3.5:7.8b"],
        "judge_model": "gpt-5"
      },
      "status": "completed",
      "completed_steps": [
        "questions",
        "classification",
        "model_gpt-4o-mini",
        "model_ollama/exaone3.5:7.8b",
        "analysis"
      ]
    }
  }
}
```

## 🚀 Usage Examples

### New Experiment
```bash
# 새 실험 시작 (자동 ID 부여)
python unified_benchmark_v2.py --questions 50 --models thesis

# 출력:
# 🆕 새 실험 생성: ID=4
# 🧪 실험 ID: 4
```

### Resume Experiment
```bash
# 중단된 실험 재개
python unified_benchmark_v2.py --resume-id 4

# 출력:
# 🔄 실험 재개: ID=4
# 📊 완료된 단계: ['questions', 'classification']
# 📂 체크포인트 로드: questions
# 🤖 모델 실행 시작...
```

## 🔄 Execution Flow

### Step 1: Questions Generation
```python
# 질문 생성 완료시
self.exp_manager.save_checkpoint(exp_id, "questions", questions_data)
self.exp_manager.add_completed_step(exp_id, "questions")
```

### Step 2: Classification
```python
# 분류 완료시
self.exp_manager.save_checkpoint(exp_id, "classification", classified_data)
self.exp_manager.add_completed_step(exp_id, "classification")
```

### Step 3: Model Evaluation (per model)
```python
# 각 모델 평가 완료시
for model in models:
    results = evaluate_model(model)
    self.exp_manager.save_checkpoint(exp_id, f"model_{model}", results)
    self.exp_manager.add_completed_step(exp_id, f"model_{model}")
```

### Step 4: Analysis
```python
# 분석 완료시
self.exp_manager.save_checkpoint(exp_id, "analysis", analysis_results)
self.exp_manager.add_completed_step(exp_id, "analysis")
self.exp_manager.update_status(exp_id, "completed")
```

## 🔧 Implementation Details

### ExperimentManager Class

```python
class ExperimentManager:
    """실험 ID 및 체크포인트 관리"""

    def create_experiment(config: dict) -> int:
        """새 실험 생성, ID 반환"""

    def get_experiment(exp_id: int) -> dict:
        """실험 정보 조회"""

    def save_checkpoint(exp_id: int, step: str, data: Any):
        """체크포인트 저장"""

    def load_checkpoint(exp_id: int, step: str) -> Any:
        """체크포인트 로드"""

    def add_completed_step(exp_id: int, step: str):
        """완료 단계 추가"""

    def update_status(exp_id: int, status: str):
        """실험 상태 업데이트"""
```

## ⚠️ Current Issues & Solutions

### Issue: RAGAS TestSet Generation Failure
- **Problem**: RAGAS library fails with "headlines property not found" error
- **Solution**: Created `unified_benchmark_v3_simple.py` with predefined questions
- **Status**: Workaround implemented, investigating RAGAS fix

### Files Created

1. **unified_benchmark_v2.py**: Full checkpoint/resume implementation (RAGAS dependent)
2. **unified_benchmark_v3_simple.py**: Simplified version with predefined questions
3. **test_checkpoint_simple.py**: Basic checkpoint functionality test
4. **test_partial_experiment.py**: Resume functionality test

## ✅ Verification

체크포인트/재개 시스템이 성공적으로 작동함을 확인:

1. ✅ 실험 ID 자동 증가
2. ✅ 메타데이터 영구 저장
3. ✅ 단계별 체크포인트 저장
4. ✅ 부분 완료 실험 재개
5. ✅ 체크포인트 데이터 복원

## 📝 Next Steps

1. RAGAS testset generation 문제 해결
2. 실제 모델 평가 통합
3. 결과 분석 및 리포트 생성 구현
4. 병렬 모델 실행 지원