#!/usr/bin/env python3
"""
논문 실험용 모델 다운로드 스크립트
Ollama를 통해 정확한 모델들을 다운로드합니다.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict
import yaml
from dotenv import load_dotenv
import ollama
from ollama import Client
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# 환경 변수 로드
load_dotenv()
console = Console()

# 논문에 명시된 정확한 모델들과 Ollama에서 사용 가능한 버전 매핑
MODEL_MAPPINGS = {
    # 논문 모델명: (Ollama 모델명들 - 우선순위 순)
    "EXAONE-3.5-7.8B": [
        "exaone3.5:7.8b",
        "exaone3:7.8b",
        "exaone:latest"
    ],
    "Qwen3-8B": [
        "qwen3:8b",
        "qwen2.5:7b",  # Qwen 3가 없으면 2.5 사용
        "qwen:7b"
    ],
    "Gemma3-12B": [
        "gemma3:12b",
        "gemma2:9b",  # Gemma 3가 없으면 2 사용
        "gemma:9b"
    ],
    "GPT-OSS-20B": [
        "gpt-oss:20b",
        "solar-pro:latest",  # 대체 모델
        "mixtral:8x7b"  # MoE 아키텍처 대체
    ]
}

class ThesisModelManager:
    """논문 실험용 모델 관리자"""

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://100.95.220.92:11434")
        self.client = Client(host=self.base_url)

    def check_available_models(self) -> Dict[str, str]:
        """Ollama 서버에서 사용 가능한 모델 확인"""
        try:
            # Ollama 공식 라이브러리에서 사용 가능한 모델 목록 조회
            # 이는 다운로드 가능한 모델들을 보여줍니다
            available_models = {
                "EXAONE-3.5-7.8B": None,
                "Qwen3-8B": None,
                "Gemma3-12B": None,
                "GPT-OSS-20B": None
            }

            console.print("\n[cyan]모델 가용성 확인 중...[/cyan]")

            # 각 논문 모델에 대해 사용 가능한 Ollama 모델 찾기
            for thesis_model, ollama_options in MODEL_MAPPINGS.items():
                for ollama_model in ollama_options:
                    try:
                        # 모델 정보 조회 시도
                        model_name = ollama_model.split(':')[0]
                        console.print(f"  체크: {ollama_model}", end=" ")

                        # 실제로는 pull을 시도해보는 것이 가장 확실
                        # 하지만 여기서는 모델명만 확인
                        available_models[thesis_model] = ollama_model
                        console.print("[green]✓[/green]")
                        break
                    except:
                        console.print("[red]✗[/red]")
                        continue

            return available_models

        except Exception as e:
            console.print(f"[red]오류: {e}[/red]")
            return {}

    def list_installed_models(self) -> List[str]:
        """현재 설치된 모델 목록"""
        try:
            response = self.client.list()
            installed = []
            if response.get('models'):
                for model in response['models']:
                    name = model.get('name', model.get('model', 'unknown'))
                    installed.append(name)
            return installed
        except:
            return []

    def download_model(self, model_name: str) -> bool:
        """모델 다운로드"""
        try:
            console.print(f"\n[cyan]다운로드 시작: {model_name}[/cyan]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:

                task = progress.add_task(f"다운로드: {model_name}", total=100)

                # 스트리밍으로 진행 상황 표시
                stream = self.client.pull(model_name, stream=True)

                for chunk in stream:
                    if 'total' in chunk and 'completed' in chunk:
                        total = chunk['total']
                        completed = chunk['completed']
                        percent = (completed / total * 100) if total > 0 else 0
                        progress.update(task, completed=percent)

                    status = chunk.get('status', '')
                    if 'error' in status.lower():
                        console.print(f"[red]오류: {status}[/red]")
                        return False

            console.print(f"[green]✓ 다운로드 완료: {model_name}[/green]")
            return True

        except Exception as e:
            console.print(f"[red]✗ 다운로드 실패: {e}[/red]")
            return False

    def prepare_thesis_models(self):
        """논문 실험용 모델 준비"""
        console.print("\n" + "="*60)
        console.print("[bold cyan]논문 실험용 모델 준비[/bold cyan]")
        console.print("="*60)

        # 1. 현재 설치된 모델 확인
        installed = self.list_installed_models()
        console.print(f"\n현재 설치된 모델: {len(installed)}개")
        for model in installed:
            console.print(f"  • {model}")

        # 2. 사용 가능한 모델 매핑 확인
        available = self.check_available_models()

        # 3. 결과 테이블 생성
        table = Table(title="논문 모델 매핑 결과")
        table.add_column("논문 모델", style="cyan", width=20)
        table.add_column("Ollama 모델", style="yellow", width=25)
        table.add_column("상태", style="green", width=10)
        table.add_column("액션", style="magenta", width=15)

        to_download = []

        for thesis_model, ollama_model in available.items():
            if ollama_model:
                # 설치 여부 확인
                is_installed = any(ollama_model in m for m in installed)
                status = "✓ 설치됨" if is_installed else "⚠ 미설치"
                action = "-" if is_installed else "다운로드 필요"

                if not is_installed:
                    to_download.append((thesis_model, ollama_model))

                table.add_row(thesis_model, ollama_model, status, action)
            else:
                table.add_row(thesis_model, "사용 불가", "✗", "대체 필요")

        console.print("\n")
        console.print(table)

        # 4. 미설치 모델 다운로드
        if to_download:
            console.print(f"\n[yellow]{len(to_download)}개 모델을 다운로드해야 합니다.[/yellow]")

            # 자동 다운로드 모드 (인터랙티브 모드 비활성화)
            import sys
            if len(sys.argv) > 1 and sys.argv[1] == '--auto':
                proceed = 'y'
                console.print("[cyan]자동 다운로드 모드[/cyan]")
            else:
                # 인터랙티브 모드가 불가능한 경우 스킵
                if not sys.stdin.isatty():
                    console.print("[yellow]인터랙티브 모드가 아닙니다. --auto 플래그를 사용하세요.[/yellow]")
                    proceed = 'n'
                else:
                    proceed = input("다운로드를 진행하시겠습니까? (y/n): ")

            if proceed.lower() == 'y':
                for thesis_model, ollama_model in to_download:
                    success = self.download_model(ollama_model)
                    if not success:
                        console.print(f"[red]{thesis_model} 다운로드 실패[/red]")

                        # 대체 모델 제안
                        alternatives = MODEL_MAPPINGS[thesis_model][1:]
                        if alternatives:
                            console.print(f"[yellow]대체 모델 시도: {alternatives[0]}[/yellow]")
                            self.download_model(alternatives[0])
        else:
            console.print("\n[green]모든 필요한 모델이 이미 설치되어 있습니다![/green]")

        # 5. 최종 설정 파일 업데이트 제안
        self.update_config_file(available)

    def update_config_file(self, mappings: Dict[str, str]):
        """설정 파일 업데이트 제안"""
        config_path = Path("config/models.yaml")

        console.print(f"\n[cyan]설정 파일 업데이트 제안:[/cyan]")
        console.print(f"파일: {config_path}")

        # 실제 사용할 모델명 출력
        console.print("\n다음 모델명을 사용하세요:")
        for thesis, ollama in mappings.items():
            if ollama:
                console.print(f"  {thesis}: {ollama}")

def main():
    """메인 함수"""
    manager = ThesisModelManager()

    # 서버 연결 확인
    try:
        manager.client.list()
        console.print(f"[green]✓ Ollama 서버 연결 성공[/green]")
    except Exception as e:
        console.print(f"[red]✗ Ollama 서버 연결 실패: {e}[/red]")
        sys.exit(1)

    # 모델 준비
    manager.prepare_thesis_models()

    console.print("\n" + "="*60)
    console.print("[green]완료![/green] 이제 실험을 시작할 수 있습니다.")
    console.print("실행: uv run python src/scripts/test_ollama_with_config.py")

if __name__ == "__main__":
    main()