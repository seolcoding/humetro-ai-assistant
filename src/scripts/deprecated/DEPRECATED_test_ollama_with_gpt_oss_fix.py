#!/usr/bin/env python3
"""
Fixed version of Ollama test script with GPT-OSS-20B prompt formatting support.
GPT-OSS-20B requires specific prompt format: <|user|>...<|assistant|>
"""

import os
import sys
import time
import json
import yaml
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

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
        config.experiment = data.get('experiment', {})
        config.metrics = data.get('metrics', [])

        return config

class OllamaModelTesterFixed:
    """Fixed Ollama 모델 테스터 with GPT-OSS support"""

    def __init__(self, config_path: Path = None):
        if config_path is None:
            config_path = Path("config/models.yaml")

        if not config_path.exists():
            raise FileNotFoundError(f"설정 파일이 없습니다: {config_path}")

        self.config = TestConfig.from_yaml(config_path)
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://100.95.220.92:11434")
        self.test_results = []

        console.print(f"[green]✓[/green] 설정 로드 완료: {config_path}")
        console.print(f"  • 활성 모델: {len(self.config.models)}개")
        console.print(f"  • 테스트 프롬프트: {len(self.config.test_prompts)}개")

    def check_connection(self) -> bool:
        """Ollama 서버 연결 확인"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                console.print(f"[green]✓[/green] Ollama 서버 연결 성공: {self.base_url}")

                models = response.json().get('models', [])
                installed_models = []

                if models:
                    console.print("\n[cyan]현재 설치된 모델:[/cyan]")
                    for model in models:
                        model_name = model.get('name', 'unknown')
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

    def format_prompt_for_model(self, model_name: str, prompt: str) -> str:
        """모델별 프롬프트 포맷팅"""
        if "gpt-oss" in model_name.lower():
            # GPT-OSS needs specific format
            return f"<|user|>\n{prompt}\n<|assistant|>"
        elif "qwen3" in model_name.lower() and "<think>" not in prompt:
            # Qwen3 sometimes includes thinking tags, handle them
            return f"User: {prompt}\nAssistant:"
        else:
            # Default format for other models
            return prompt

    def generate_response(self, model_name: str, prompt: str, options: Dict) -> Dict:
        """Generate response using Ollama API directly"""
        formatted_prompt = self.format_prompt_for_model(model_name, prompt)

        payload = {
            "model": model_name,
            "prompt": formatted_prompt,
            "stream": False,
            "options": options
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=options.get('timeout', 120)
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

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
            console.print(f"\n[cyan]테스트 시작:[/cyan] {model_config.name}")

            # Check if model is installed
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            installed = [m.get('name', '') for m in response.json().get('models', [])]

            if not any(model_config.ollama_name in name for name in installed):
                console.print(f"[yellow]모델 미설치: {model_config.ollama_name}[/yellow]")
                result['error'] = "모델 미설치"
                return result

            # Warmup
            console.print("  웜업 중...")
            warmup_start = time.time()
            warmup_response = self.generate_response(
                model_config.ollama_name,
                "Hello",
                {"temperature": 0.1, "num_predict": 10}
            )
            result['load_time'] = time.time() - warmup_start
            console.print(f"  [green]✓[/green] 모델 로드: {result['load_time']:.2f}초")

            # Test prompts
            total_tokens = 0
            total_time = 0

            for i, prompt_data in enumerate(self.config.test_prompts, 1):
                prompt = prompt_data['prompt']
                category = prompt_data.get('category', 'general')

                console.print(f"  테스트 {i}/{len(self.config.test_prompts)}: [{category}]")

                options = {
                    "temperature": self.config.experiment.get('temperature', 0.1),
                    "num_predict": self.config.experiment.get('max_tokens', 256),
                    "top_p": self.config.experiment.get('top_p', 0.9),
                    "repeat_penalty": self.config.experiment.get('repeat_penalty', 1.1),
                    "seed": 42  # for reproducibility
                }

                start_time = time.time()
                response_data = self.generate_response(
                    model_config.ollama_name,
                    prompt,
                    options
                )
                inference_time = time.time() - start_time

                if "error" in response_data:
                    console.print(f"    [red]오류: {response_data['error']}[/red]")
                    response_text = ""
                else:
                    response_text = response_data.get('response', '')

                # Save response
                result['responses'].append({
                    "prompt": prompt,
                    "category": category,
                    "response": response_text,
                    "inference_time": inference_time,
                    "response_length": len(response_text)
                })

                total_time += inference_time
                if response_text:
                    total_tokens += len(response_text.split())

            # Calculate metrics
            if result['responses']:
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
        console.print("[bold cyan]Ollama 모델 테스트 시작 (GPT-OSS Fix)[/bold cyan]")
        console.print("="*60)

        for model_config in self.config.models:
            result = self.test_model(model_config)
            self.test_results.append(result)
            time.sleep(2)

        self.save_results()
        self.print_results()

    def save_results(self):
        """테스트 결과 저장"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = results_dir / f"ollama_test_fixed_{timestamp}.json"

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

        successful_results = [r for r in self.test_results if r['status'] == 'success']

        if not successful_results:
            console.print("[red]성공한 테스트가 없습니다.[/red]")
            return

        # Results table
        table = Table(title="모델 성능 비교 (Fixed)")
        table.add_column("모델", style="cyan", width=20)
        table.add_column("크기", style="yellow", width=8)
        table.add_column("로드(초)", style="green", justify="right", width=8)
        table.add_column("평균응답(초)", style="green", justify="right", width=10)
        table.add_column("TPS", style="blue", justify="right", width=8)
        table.add_column("평균길이", style="magenta", justify="right", width=10)

        for result in successful_results:
            table.add_row(
                result['model'],
                result['size'],
                f"{result['load_time']:.2f}",
                f"{result['metrics']['avg_response_time']:.2f}",
                f"{result['metrics']['tokens_per_second']:.1f}",
                f"{result['metrics']['avg_response_length']:.0f}"
            )

        console.print(table)

        # Best performing models
        if successful_results:
            valid_results = [r for r in successful_results if r['metrics']['avg_response_length'] > 0]

            if valid_results:
                best_speed = min(valid_results, key=lambda x: x['metrics']['avg_response_time'])
                best_tps = max(valid_results, key=lambda x: x['metrics']['tokens_per_second'])

                console.print(f"\n[green]🏆 최고 성능:[/green]")
                console.print(f"  • 가장 빠른 응답: {best_speed['model']} ({best_speed['metrics']['avg_response_time']:.2f}초)")
                console.print(f"  • 최고 처리량: {best_tps['model']} ({best_tps['metrics']['tokens_per_second']:.1f} TPS)")

                # Check for empty responses
                empty_responses = [r for r in successful_results if r['metrics']['avg_response_length'] == 0]
                if empty_responses:
                    console.print(f"\n[yellow]⚠️ 빈 응답 모델:[/yellow]")
                    for r in empty_responses:
                        console.print(f"  • {r['model']} - 프롬프트 포맷 확인 필요")

def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="Ollama 모델 테스트 (GPT-OSS Fixed)")
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

    try:
        tester = OllamaModelTesterFixed(args.config)
    except FileNotFoundError as e:
        console.print(f"[red]오류: {e}[/red]")
        sys.exit(1)

    if not tester.check_connection():
        console.print("[red]서버에 연결할 수 없습니다.[/red]")
        sys.exit(1)

    if args.check_only:
        console.print("\n[green]연결 확인 완료[/green]")
        sys.exit(0)

    try:
        tester.run_tests()
    except KeyboardInterrupt:
        console.print("\n[yellow]테스트가 중단되었습니다.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]테스트 중 오류 발생: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()