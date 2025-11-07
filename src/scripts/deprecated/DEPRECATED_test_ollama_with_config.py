#!/usr/bin/env python3
"""
설정 파일 기반 Ollama 모델 테스트 스크립트
외부 YAML 설정 파일에서 모델 목록과 테스트 설정을 읽어옵니다.
"""

import os
import sys
import time
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

import ollama
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

# 환경 변수 로드
load_dotenv()

console = Console()

@dataclass
class ModelConfig:
    """모델 설정 클래스"""
    name: str
    ollama_name: str
    size: str
    quantization: Optional[str] = None
    description: str = ""
    category: str = "general"
    enabled: bool = True
    vram_usage: Optional[str] = None
    tokens_per_sec: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelConfig':
        """딕셔너리에서 ModelConfig 생성"""
        # 알려진 필드만 추출
        known_fields = {
            'name', 'ollama_name', 'size', 'quantization',
            'description', 'category', 'enabled', 'vram_usage', 'tokens_per_sec'
        }
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)

@dataclass
class TestConfig:
    """테스트 설정 클래스"""
    models: List[ModelConfig] = field(default_factory=list)
    test_prompts: List[Dict] = field(default_factory=list)
    experiment: Dict = field(default_factory=dict)
    metrics: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'TestConfig':
        """YAML 파일에서 TestConfig 생성"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        config = cls()

        # 모델 로드 (enabled=True인 것만)
        for model_data in data.get('models', []):
            if model_data.get('enabled', True):
                config.models.append(ModelConfig.from_dict(model_data))

        # 테스트 프롬프트 로드
        config.test_prompts = data.get('test_prompts', [])

        # 실험 설정 로드
        config.experiment = data.get('experiment', {})

        # 메트릭 로드
        config.metrics = data.get('metrics', [])

        return config

class OllamaModelTester:
    """Ollama 모델 테스터 클래스"""

    def __init__(self, config_path: Path = None):
        # 기본 설정 경로
        if config_path is None:
            config_path = Path("config/models.yaml")

        # 설정 로드
        if not config_path.exists():
            raise FileNotFoundError(f"설정 파일이 없습니다: {config_path}")

        self.config = TestConfig.from_yaml(config_path)
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://100.95.220.92:11434")
        self.client = ollama.Client(host=self.base_url)
        self.test_results = []

        console.print(f"[green]✓[/green] 설정 로드 완료: {config_path}")
        console.print(f"  • 활성 모델: {len(self.config.models)}개")
        console.print(f"  • 테스트 프롬프트: {len(self.config.test_prompts)}개")

    def check_connection(self) -> bool:
        """Ollama 서버 연결 확인"""
        try:
            response = self.client.list()
            console.print(f"[green]✓[/green] Ollama 서버 연결 성공: {self.base_url}")

            # 현재 설치된 모델 목록
            installed_models = []
            if response.get('models'):
                console.print("\n[cyan]현재 설치된 모델:[/cyan]")
                for model in response['models']:
                    model_name = model.get('name', model.get('model', 'unknown'))
                    size_gb = model.get('size', 0) / (1024**3)
                    console.print(f"  • {model_name} ({size_gb:.2f}GB)")
                    installed_models.append(model_name)

            # 설정된 모델과 비교
            console.print("\n[cyan]설정된 모델 상태:[/cyan]")
            for model in self.config.models:
                is_installed = any(model.ollama_name in m for m in installed_models)
                status = "[green]✓ 설치됨[/green]" if is_installed else "[yellow]⚠ 미설치[/yellow]"
                console.print(f"  • {model.name} ({model.ollama_name}): {status}")

            return True

        except Exception as e:
            console.print(f"[red]✗[/red] 서버 연결 실패: {e}")
            return False

    def pull_model_with_progress(self, model_name: str) -> bool:
        """진행 상황 표시와 함께 모델 다운로드"""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console,
            ) as progress:

                task = progress.add_task(f"[cyan]모델 다운로드: {model_name}", total=100)

                stream = self.client.pull(model_name, stream=True)

                for chunk in stream:
                    if 'total' in chunk and 'completed' in chunk:
                        total = chunk['total']
                        completed = chunk['completed']
                        percent = (completed / total * 100) if total > 0 else 0
                        progress.update(task, completed=percent)

            console.print(f"[green]✓[/green] 모델 다운로드 완료: {model_name}")
            return True

        except Exception as e:
            console.print(f"[red]✗[/red] 모델 다운로드 실패: {e}")
            return False

    def test_model(self, model_config: ModelConfig) -> Dict[str, Any]:
        """개별 모델 테스트"""
        result = {
            "model": model_config.name,
            "ollama_name": model_config.ollama_name,
            "size": model_config.size,
            "category": model_config.category,
            "status": "failed",
            "load_time": 0,
            "responses": [],
            "metrics": {},
            "error": None,
            "timestamp": datetime.now().isoformat()
        }

        try:
            # 모델 설치 확인
            models = self.client.list()
            installed = [m.get('name', m.get('model', '')) for m in models.get('models', [])]

            if not any(model_config.ollama_name in name for name in installed):
                console.print(f"[yellow]모델 미설치: {model_config.ollama_name}[/yellow]")
                if not self.pull_model_with_progress(model_config.ollama_name):
                    result['error'] = "모델 다운로드 실패"
                    return result

            console.print(f"\n[cyan]테스트 시작:[/cyan] {model_config.name}")

            # LangChain LLM 초기화
            llm = OllamaLLM(
                base_url=self.base_url,
                model=model_config.ollama_name,
                temperature=self.config.experiment.get('temperature', 0.1),
                num_predict=self.config.experiment.get('max_tokens', 256),
                top_p=self.config.experiment.get('top_p', 0.9),
                repeat_penalty=self.config.experiment.get('repeat_penalty', 1.1),
            )

            # 웜업
            start_time = time.time()
            _ = llm.invoke("시스템 준비")
            result['load_time'] = time.time() - start_time
            console.print(f"  [green]✓[/green] 모델 로드: {result['load_time']:.2f}초")

            # 테스트 프롬프트 실행
            total_tokens = 0
            total_time = 0

            for i, prompt_data in enumerate(self.config.test_prompts, 1):
                prompt = prompt_data['prompt']
                category = prompt_data.get('category', 'general')

                console.print(f"  테스트 {i}/{len(self.config.test_prompts)}: [{category}]")

                start_time = time.time()
                response = llm.invoke(prompt)
                inference_time = time.time() - start_time

                # 응답 저장
                result['responses'].append({
                    "prompt": prompt,
                    "category": category,
                    "response": response,
                    "inference_time": inference_time,
                    "response_length": len(response)
                })

                total_time += inference_time
                total_tokens += len(response.split())  # 간단한 토큰 추정

            # 메트릭 계산
            result['metrics'] = {
                "avg_response_time": total_time / len(self.config.test_prompts),
                "total_time": total_time,
                "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
                "avg_response_length": sum(r['response_length'] for r in result['responses']) / len(result['responses'])
            }

            result['status'] = "success"
            console.print(f"  [green]✓[/green] 테스트 완료")

        except Exception as e:
            result['error'] = str(e)
            console.print(f"  [red]✗[/red] 테스트 실패: {e}")

        return result

    def run_tests(self):
        """모든 모델 테스트 실행"""
        console.print("\n" + "="*60)
        console.print("[bold cyan]Ollama 모델 테스트 시작[/bold cyan]")
        console.print("="*60)

        for model_config in self.config.models:
            result = self.test_model(model_config)
            self.test_results.append(result)
            time.sleep(2)  # 모델 간 전환 대기

        self.save_results()
        self.print_results()

    def save_results(self):
        """테스트 결과 저장"""
        # 결과 디렉토리 생성
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # 타임스탬프 파일명
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = results_dir / f"ollama_test_{timestamp}.json"

        # JSON 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "config": {
                    "base_url": self.base_url,
                    "experiment": self.config.experiment,
                    "metrics": self.config.metrics
                },
                "results": self.test_results
            }, f, ensure_ascii=False, indent=2)

        console.print(f"\n[green]결과 저장:[/green] {output_file}")

    def print_results(self):
        """테스트 결과 출력"""
        console.print("\n" + "="*60)
        console.print("[bold cyan]테스트 결과 요약[/bold cyan]")
        console.print("="*60)

        # 성공한 테스트만 필터링
        successful_results = [r for r in self.test_results if r['status'] == 'success']

        if not successful_results:
            console.print("[red]성공한 테스트가 없습니다.[/red]")
            return

        # 결과 테이블
        table = Table(title="모델 성능 비교")
        table.add_column("모델", style="cyan", width=20)
        table.add_column("크기", style="yellow", width=8)
        table.add_column("카테고리", style="magenta", width=12)
        table.add_column("로드(초)", style="green", justify="right", width=8)
        table.add_column("평균응답(초)", style="green", justify="right", width=10)
        table.add_column("TPS", style="blue", justify="right", width=8)

        for result in successful_results:
            table.add_row(
                result['model'],
                result['size'],
                result['category'],
                f"{result['load_time']:.2f}",
                f"{result['metrics']['avg_response_time']:.2f}",
                f"{result['metrics']['tokens_per_second']:.1f}"
            )

        console.print(table)

        # 최고 성능 모델
        if successful_results:
            best_speed = min(successful_results, key=lambda x: x['metrics']['avg_response_time'])
            best_tps = max(successful_results, key=lambda x: x['metrics']['tokens_per_second'])

            console.print(f"\n[green]🏆 최고 성능:[/green]")
            console.print(f"  • 가장 빠른 응답: {best_speed['model']} ({best_speed['metrics']['avg_response_time']:.2f}초)")
            console.print(f"  • 최고 처리량: {best_tps['model']} ({best_tps['metrics']['tokens_per_second']:.1f} TPS)")

def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="Ollama 모델 테스트")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/models.yaml"),
        help="모델 설정 파일 경로"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="연결 확인만 수행"
    )

    args = parser.parse_args()

    # 테스터 초기화
    try:
        tester = OllamaModelTester(args.config)
    except FileNotFoundError as e:
        console.print(f"[red]오류: {e}[/red]")
        sys.exit(1)

    # 서버 연결 확인
    if not tester.check_connection():
        console.print("[red]서버에 연결할 수 없습니다.[/red]")
        sys.exit(1)

    # 연결 확인만 수행
    if args.check_only:
        console.print("\n[green]연결 확인 완료[/green]")
        sys.exit(0)

    # 테스트 실행
    try:
        tester.run_tests()
    except KeyboardInterrupt:
        console.print("\n[yellow]테스트가 중단되었습니다.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]테스트 중 오류 발생: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()