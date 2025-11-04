#!/usr/bin/env python3
"""
논문 실험 모델 검증 스크립트
모든 4개 모델이 정상 작동하는지 확인하고 성능 기준선을 설정합니다.
"""

import os
import sys
import time
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

import ollama
from ollama import Client
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# 환경 변수 로드
load_dotenv()
console = Console()

# 논문 실험용 테스트 프롬프트
THESIS_TEST_PROMPTS = [
    {
        "prompt": "서울시 120 다산콜센터는 무엇인가요?",
        "category": "정보_질의",
        "expected_keywords": ["120", "민원", "상담", "서울시", "콜센터"],
    },
    {
        "prompt": "코로나19 백신 접종 예약은 어떻게 하나요?",
        "category": "절차_안내",
        "expected_keywords": ["예약", "백신", "접종", "방법"],
    },
    {
        "prompt": "주민등록등본을 온라인으로 발급받는 방법을 알려주세요.",
        "category": "민원_처리",
        "expected_keywords": ["정부24", "온라인", "발급", "주민등록등본"],
    },
    {
        "prompt": "지하철 운행 시간과 막차 시간을 알려주세요.",
        "category": "교통_정보",
        "expected_keywords": ["지하철", "운행", "시간", "막차"],
    },
    {
        "prompt": "재난지원금 신청 자격과 방법은 어떻게 되나요?",
        "category": "복지_지원",
        "expected_keywords": ["재난지원금", "신청", "자격", "방법"],
    }
]

@dataclass
class ModelValidationResult:
    """모델 검증 결과 클래스"""
    model_name: str
    ollama_name: str
    size: str
    quantization: Optional[str]

    # 가용성
    is_installed: bool = False
    installation_error: Optional[str] = None

    # 성능 메트릭
    load_time: float = 0.0
    avg_response_time: float = 0.0
    tokens_per_second: float = 0.0
    memory_usage_gb: Optional[float] = None

    # 품질 메트릭
    response_quality: Dict[str, Any] = field(default_factory=dict)
    keyword_coverage: float = 0.0  # 예상 키워드 포함 비율

    # 응답 샘플
    response_samples: List[Dict] = field(default_factory=list)

    # 검증 상태
    validation_status: str = "pending"  # pending, passed, failed, error
    validation_errors: List[str] = field(default_factory=list)

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class ThesisModelValidator:
    """논문 실험 모델 검증기"""

    def __init__(self, config_path: Path = None):
        if config_path is None:
            config_path = Path("config/models.yaml")

        if not config_path.exists():
            raise FileNotFoundError(f"설정 파일이 없습니다: {config_path}")

        # 설정 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://100.95.220.92:11434")
        self.client = Client(host=self.base_url)
        self.results: List[ModelValidationResult] = []

        console.print(f"[green]✓[/green] 검증 환경 설정 완료")
        console.print(f"  • Ollama 서버: {self.base_url}")
        console.print(f"  • 검증할 모델: {len([m for m in self.config['models'] if m.get('enabled', True)])}개")
        console.print(f"  • 테스트 프롬프트: {len(THESIS_TEST_PROMPTS)}개")

    def check_model_availability(self, model_config: Dict) -> ModelValidationResult:
        """모델 가용성 확인"""
        result = ModelValidationResult(
            model_name=model_config['name'],
            ollama_name=model_config['ollama_name'],
            size=model_config['size'],
            quantization=model_config.get('quantization')
        )

        try:
            # 설치된 모델 확인
            models = self.client.list()
            installed_models = [m.get('name', '') for m in models.get('models', [])]

            result.is_installed = any(
                model_config['ollama_name'] in m for m in installed_models
            )

            if not result.is_installed:
                result.validation_status = "failed"
                result.validation_errors.append(f"모델이 설치되지 않음: {model_config['ollama_name']}")

        except Exception as e:
            result.installation_error = str(e)
            result.validation_status = "error"
            result.validation_errors.append(f"가용성 확인 실패: {e}")

        return result

    def test_model_performance(self, model_config: Dict, result: ModelValidationResult):
        """모델 성능 테스트"""
        try:
            console.print(f"\n[cyan]성능 테스트 시작:[/cyan] {model_config['name']}")

            # LangChain LLM 초기화
            llm = OllamaLLM(
                base_url=self.base_url,
                model=model_config['ollama_name'],
                temperature=0.1,
                num_predict=256,
                top_p=0.9,
            )

            # 1. 모델 로드 시간 측정 (웜업)
            start_time = time.time()
            _ = llm.invoke("시스템 준비")
            result.load_time = time.time() - start_time
            console.print(f"  [green]✓[/green] 모델 로드: {result.load_time:.2f}초")

            # 2. 응답 시간 및 품질 테스트
            total_time = 0
            total_tokens = 0
            keyword_matches = 0
            total_keywords = 0

            for i, test_case in enumerate(THESIS_TEST_PROMPTS, 1):
                console.print(f"  테스트 {i}/{len(THESIS_TEST_PROMPTS)}: {test_case['category']}")

                start_time = time.time()
                response = llm.invoke(test_case['prompt'])
                response_time = time.time() - start_time

                # 메트릭 계산
                response_tokens = len(response.split())
                total_time += response_time
                total_tokens += response_tokens

                # 키워드 커버리지 체크
                response_lower = response.lower()
                matched_keywords = [
                    kw for kw in test_case['expected_keywords']
                    if kw.lower() in response_lower
                ]
                keyword_matches += len(matched_keywords)
                total_keywords += len(test_case['expected_keywords'])

                # 응답 샘플 저장
                result.response_samples.append({
                    "prompt": test_case['prompt'],
                    "category": test_case['category'],
                    "response": response[:500],  # 처음 500자만 저장
                    "response_time": response_time,
                    "tokens": response_tokens,
                    "keyword_coverage": len(matched_keywords) / len(test_case['expected_keywords'])
                })

            # 3. 평균 메트릭 계산
            result.avg_response_time = total_time / len(THESIS_TEST_PROMPTS)
            result.tokens_per_second = total_tokens / total_time if total_time > 0 else 0
            result.keyword_coverage = keyword_matches / total_keywords if total_keywords > 0 else 0

            # 4. 품질 평가
            result.response_quality = {
                "avg_response_length": sum(s['tokens'] for s in result.response_samples) / len(result.response_samples),
                "keyword_coverage_percent": result.keyword_coverage * 100,
                "consistency_score": self._calculate_consistency(result.response_samples),
            }

            # 5. 검증 상태 결정
            if result.keyword_coverage >= 0.3 and result.avg_response_time < 30:
                result.validation_status = "passed"
                console.print(f"  [green]✓[/green] 성능 테스트 통과")
            else:
                result.validation_status = "failed"
                if result.keyword_coverage < 0.3:
                    result.validation_errors.append(f"키워드 커버리지 부족: {result.keyword_coverage:.1%}")
                if result.avg_response_time >= 30:
                    result.validation_errors.append(f"응답 시간 초과: {result.avg_response_time:.1f}초")
                console.print(f"  [yellow]⚠[/yellow] 성능 기준 미달")

        except Exception as e:
            result.validation_status = "error"
            result.validation_errors.append(f"성능 테스트 실패: {e}")
            console.print(f"  [red]✗[/red] 테스트 실패: {e}")

    def _calculate_consistency(self, samples: List[Dict]) -> float:
        """응답 일관성 점수 계산"""
        if len(samples) < 2:
            return 1.0

        # 응답 시간의 변동계수(CV) 기반 일관성
        response_times = [s['response_time'] for s in samples]
        mean_time = sum(response_times) / len(response_times)
        if mean_time == 0:
            return 1.0

        std_dev = (sum((t - mean_time) ** 2 for t in response_times) / len(response_times)) ** 0.5
        cv = std_dev / mean_time

        # CV가 낮을수록 일관성이 높음 (0~1 scale)
        consistency = max(0, 1 - cv)
        return consistency

    def validate_all_models(self):
        """모든 모델 검증 실행"""
        console.print("\n" + "="*60)
        console.print("[bold cyan]논문 실험 모델 검증 시작[/bold cyan]")
        console.print("="*60)

        # 활성화된 모델만 검증
        enabled_models = [
            m for m in self.config['models']
            if m.get('enabled', True)
        ]

        for model_config in enabled_models:
            # 1. 가용성 확인
            result = self.check_model_availability(model_config)

            # 2. 설치되어 있으면 성능 테스트
            if result.is_installed:
                self.test_model_performance(model_config, result)
            else:
                console.print(f"\n[yellow]⚠ {model_config['name']}: 모델 미설치로 테스트 건너뜀[/yellow]")

            self.results.append(result)
            time.sleep(2)  # 모델 간 전환 대기

    def generate_report(self):
        """검증 보고서 생성"""
        console.print("\n" + "="*60)
        console.print("[bold cyan]검증 결과 보고서[/bold cyan]")
        console.print("="*60)

        # 1. 요약 테이블
        table = Table(title="모델 검증 요약")
        table.add_column("모델", style="cyan", width=20)
        table.add_column("크기", style="yellow", width=8)
        table.add_column("상태", style="green", width=10)
        table.add_column("로드(초)", justify="right", width=8)
        table.add_column("응답(초)", justify="right", width=8)
        table.add_column("TPS", justify="right", width=8)
        table.add_column("키워드", justify="right", width=8)

        for result in self.results:
            status_emoji = {
                "passed": "✅",
                "failed": "⚠️",
                "error": "❌",
                "pending": "⏳"
            }.get(result.validation_status, "❓")

            table.add_row(
                result.model_name,
                result.size,
                status_emoji,
                f"{result.load_time:.2f}" if result.load_time > 0 else "-",
                f"{result.avg_response_time:.2f}" if result.avg_response_time > 0 else "-",
                f"{result.tokens_per_second:.1f}" if result.tokens_per_second > 0 else "-",
                f"{result.keyword_coverage:.0%}" if result.keyword_coverage > 0 else "-"
            )

        console.print(table)

        # 2. 실험 준비 상태
        passed_models = [r for r in self.results if r.validation_status == "passed"]
        failed_models = [r for r in self.results if r.validation_status in ["failed", "error"]]

        console.print(f"\n[bold]실험 준비 상태:[/bold]")
        console.print(f"  • 통과: {len(passed_models)}/{len(self.results)} 모델")
        console.print(f"  • 실패: {len(failed_models)}/{len(self.results)} 모델")

        if passed_models:
            console.print(f"\n[green]사용 가능한 모델:[/green]")
            for model in passed_models:
                console.print(f"  ✅ {model.model_name}")

        if failed_models:
            console.print(f"\n[yellow]문제가 있는 모델:[/yellow]")
            for model in failed_models:
                console.print(f"  ⚠️ {model.model_name}: {', '.join(model.validation_errors)}")

        # 3. 권장사항
        console.print(f"\n[bold]권장사항:[/bold]")
        if len(passed_models) == 4:
            console.print("  ✅ 모든 모델이 준비되었습니다. 실험을 시작할 수 있습니다.")
        elif len(passed_models) >= 2:
            console.print("  ⚠️ 일부 모델만 사용 가능합니다. 부분 실험 가능합니다.")
        else:
            console.print("  ❌ 충분한 모델이 준비되지 않았습니다. 모델 설치를 확인하세요.")

        # 4. 상세 결과 저장
        self._save_detailed_report()

    def _save_detailed_report(self):
        """상세 검증 결과 저장"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = results_dir / f"model_validation_{timestamp}.json"

        report_data = {
            "validation_time": datetime.now().isoformat(),
            "server_url": self.base_url,
            "test_prompts": len(THESIS_TEST_PROMPTS),
            "results": [asdict(r) for r in self.results],
            "summary": {
                "total_models": len(self.results),
                "passed": len([r for r in self.results if r.validation_status == "passed"]),
                "failed": len([r for r in self.results if r.validation_status in ["failed", "error"]]),
                "avg_load_time": sum(r.load_time for r in self.results) / len(self.results) if self.results else 0,
                "avg_response_time": sum(r.avg_response_time for r in self.results) / len(self.results) if self.results else 0,
            }
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        console.print(f"\n[green]상세 보고서 저장:[/green] {report_file}")

def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="논문 실험 모델 검증")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/models.yaml"),
        help="모델 설정 파일 경로"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="빠른 검증 (첫 2개 프롬프트만 사용)"
    )

    args = parser.parse_args()

    # 빠른 검증 모드
    if args.quick:
        global THESIS_TEST_PROMPTS
        THESIS_TEST_PROMPTS = THESIS_TEST_PROMPTS[:2]
        console.print("[yellow]빠른 검증 모드: 2개 프롬프트만 사용[/yellow]")

    try:
        # 검증기 초기화
        validator = ThesisModelValidator(args.config)

        # 서버 연결 확인
        validator.client.list()
        console.print(f"[green]✓ Ollama 서버 연결 성공[/green]")

        # 모든 모델 검증
        validator.validate_all_models()

        # 보고서 생성
        validator.generate_report()

    except Exception as e:
        console.print(f"[red]검증 실패: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()