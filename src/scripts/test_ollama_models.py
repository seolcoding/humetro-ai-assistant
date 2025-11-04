#!/usr/bin/env python3
"""
Ollama 모델 테스트 스크립트
RTX 3090 서버에서 실행 중인 Ollama 엔드포인트를 통해 오픈소스 LLM 모델들을 테스트합니다.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio

import ollama
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# 환경 변수 로드
load_dotenv()

console = Console()

@dataclass
class ModelConfig:
    """모델 설정 클래스"""
    name: str
    ollama_name: str  # Ollama에서 사용하는 실제 모델 이름
    size: str  # 모델 크기
    quantization: Optional[str] = None  # 양자화 정보
    description: str = ""

# 테스트할 모델 목록
MODELS = [
    ModelConfig(
        name="EXAONE-3.5-7.8B",
        ollama_name="exaone3.5:8b",  # Ollama에서 제공하는 EXAONE 모델
        size="7.8B",
        quantization="Q4_K_M",
        description="LG AI Research의 한국어 특화 모델"
    ),
    ModelConfig(
        name="Qwen3-8B",
        ollama_name="qwen2.5:7b",  # Qwen 2.5가 최신 버전
        size="7B",
        quantization="Q4_K_M",
        description="Alibaba의 다국어 지원 모델"
    ),
    ModelConfig(
        name="Gemma3-9B",
        ollama_name="gemma2:9b",  # Gemma2가 최신
        size="9B",
        quantization="Q4_K_M",
        description="Google의 경량화 모델"
    ),
    ModelConfig(
        name="Llama3.3-70B",
        ollama_name="llama3.3:70b-instruct-q2_K",  # 70B는 양자화 필수
        size="70B",
        quantization="Q2_K",  # 24GB VRAM에 맞추기 위해 더 높은 압축
        description="Meta의 최신 대규모 모델 (높은 양자화)"
    ),
]

# 대체 모델 (위 모델이 없을 경우)
FALLBACK_MODELS = [
    ModelConfig(
        name="Mistral-7B",
        ollama_name="mistral:7b",
        size="7B",
        quantization="Q4_K_M",
        description="Mistral AI의 효율적인 모델"
    ),
    ModelConfig(
        name="Phi-3.5",
        ollama_name="phi3.5:3.8b",
        size="3.8B",
        quantization="Q4_K_M",
        description="Microsoft의 소형 고성능 모델"
    ),
]

class OllamaModelTester:
    """Ollama 모델 테스터 클래스"""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://100.95.220.92:11434")
        self.client = ollama.Client(host=self.base_url)
        self.test_results = []

    def check_connection(self) -> bool:
        """Ollama 서버 연결 확인"""
        try:
            # 서버 상태 확인
            response = self.client.list()
            console.print(f"[green]✓[/green] Ollama 서버 연결 성공: {self.base_url}")

            # 현재 설치된 모델 목록
            if response.get('models'):
                console.print("\n[cyan]현재 설치된 모델:[/cyan]")
                for model in response['models']:
                    size_gb = model.get('size', 0) / (1024**3)
                    console.print(f"  • {model['name']} ({size_gb:.2f}GB)")
            else:
                console.print("[yellow]설치된 모델이 없습니다.[/yellow]")

            return True
        except Exception as e:
            console.print(f"[red]✗[/red] 서버 연결 실패: {e}")
            return False

    def pull_model(self, model_name: str) -> bool:
        """모델 다운로드/업데이트"""
        try:
            console.print(f"\n[cyan]모델 다운로드 중:[/cyan] {model_name}")

            # 스트리밍 방식으로 진행 상황 표시
            stream = self.client.pull(model_name, stream=True)

            for chunk in stream:
                status = chunk.get('status', '')
                if 'total' in chunk and 'completed' in chunk:
                    total = chunk['total']
                    completed = chunk['completed']
                    percent = (completed / total * 100) if total > 0 else 0
                    console.print(f"  {status}: {percent:.1f}%", end='\r')
                else:
                    console.print(f"  {status}", end='\r')

            console.print(f"[green]✓[/green] 모델 다운로드 완료: {model_name}      ")
            return True

        except Exception as e:
            console.print(f"[red]✗[/red] 모델 다운로드 실패: {e}")
            return False

    def test_model(self, model_config: ModelConfig, test_prompts: List[str]) -> Dict[str, Any]:
        """개별 모델 테스트"""
        result = {
            "model": model_config.name,
            "ollama_name": model_config.ollama_name,
            "size": model_config.size,
            "status": "failed",
            "load_time": 0,
            "inference_time": 0,
            "responses": [],
            "error": None
        }

        try:
            # 모델이 설치되어 있는지 확인
            models = self.client.list()
            model_names = [m['name'] for m in models.get('models', [])]

            if not any(model_config.ollama_name in name for name in model_names):
                console.print(f"[yellow]모델이 설치되지 않음. 다운로드 시작...[/yellow]")
                if not self.pull_model(model_config.ollama_name):
                    result['error'] = "모델 다운로드 실패"
                    return result

            # 모델 로드 및 첫 번째 쿼리 (웜업)
            console.print(f"\n[cyan]테스트 중:[/cyan] {model_config.name}")

            start_time = time.time()

            # LangChain을 사용한 테스트
            llm = OllamaLLM(
                base_url=self.base_url,
                model=model_config.ollama_name,
                temperature=0.1,
                num_predict=100,  # 최대 토큰 수 제한
            )

            # 웜업 쿼리
            _ = llm.invoke("안녕하세요")
            load_time = time.time() - start_time
            result['load_time'] = load_time

            console.print(f"  [green]✓[/green] 모델 로드 완료 ({load_time:.2f}초)")

            # 테스트 프롬프트 실행
            for i, prompt in enumerate(test_prompts, 1):
                console.print(f"  테스트 {i}/{len(test_prompts)}: {prompt[:50]}...")

                start_time = time.time()
                response = llm.invoke(prompt)
                inference_time = time.time() - start_time

                result['responses'].append({
                    "prompt": prompt,
                    "response": response[:200],  # 처음 200자만 저장
                    "inference_time": inference_time
                })

                result['inference_time'] += inference_time

            result['status'] = "success"
            console.print(f"  [green]✓[/green] 테스트 완료")

        except Exception as e:
            result['error'] = str(e)
            console.print(f"  [red]✗[/red] 테스트 실패: {e}")

        return result

    def run_tests(self, models: List[ModelConfig] = None):
        """모든 모델 테스트 실행"""
        if models is None:
            models = MODELS

        # 테스트 프롬프트
        test_prompts = [
            "서울시 120 다산콜센터는 무엇인가요?",
            "코로나19 백신 접종 예약은 어떻게 하나요?",
            "주민등록등본을 온라인으로 발급받는 방법을 알려주세요."
        ]

        console.print("\n" + "="*60)
        console.print("[bold cyan]Ollama 모델 테스트 시작[/bold cyan]")
        console.print("="*60)

        for model_config in models:
            result = self.test_model(model_config, test_prompts)
            self.test_results.append(result)
            time.sleep(2)  # 모델 간 전환 시 대기

    def print_results(self):
        """테스트 결과 출력"""
        console.print("\n" + "="*60)
        console.print("[bold cyan]테스트 결과 요약[/bold cyan]")
        console.print("="*60)

        # 결과 테이블 생성
        table = Table(title="모델 성능 비교")
        table.add_column("모델", style="cyan")
        table.add_column("크기", style="yellow")
        table.add_column("상태", style="green")
        table.add_column("로드 시간(초)", style="magenta")
        table.add_column("평균 추론 시간(초)", style="magenta")

        for result in self.test_results:
            status_emoji = "✅" if result['status'] == "success" else "❌"
            avg_inference = result['inference_time'] / len(result['responses']) if result['responses'] else 0

            table.add_row(
                result['model'],
                result['size'],
                status_emoji,
                f"{result['load_time']:.2f}",
                f"{avg_inference:.2f}"
            )

        console.print(table)

        # 상세 결과 저장
        output_file = Path("ollama_test_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)

        console.print(f"\n[green]상세 결과가 {output_file}에 저장되었습니다.[/green]")

def main():
    """메인 함수"""
    # Ollama 테스터 초기화
    tester = OllamaModelTester()

    # 서버 연결 확인
    if not tester.check_connection():
        console.print("[red]서버에 연결할 수 없습니다. 설정을 확인하세요.[/red]")
        sys.exit(1)

    # 사용자에게 모델 선택 옵션 제공
    console.print("\n[cyan]테스트할 모델을 선택하세요:[/cyan]")
    console.print("1. 논문 모델 (EXAONE, Qwen, Gemma, Llama)")
    console.print("2. 대체 모델 (Mistral, Phi)")
    console.print("3. 모두 테스트")

    choice = input("\n선택 (1/2/3): ").strip()

    if choice == "1":
        models_to_test = MODELS
    elif choice == "2":
        models_to_test = FALLBACK_MODELS
    elif choice == "3":
        models_to_test = MODELS + FALLBACK_MODELS
    else:
        console.print("[yellow]기본 옵션(논문 모델)으로 진행합니다.[/yellow]")
        models_to_test = MODELS

    # 테스트 실행
    tester.run_tests(models_to_test)

    # 결과 출력
    tester.print_results()

if __name__ == "__main__":
    main()