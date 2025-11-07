#!/usr/bin/env python3
"""
간단한 Ollama 연결 테스트 스크립트
"""

import os
from dotenv import load_dotenv
import ollama
from ollama import Client

# 환경 변수 로드
load_dotenv()

def test_ollama_connection():
    """Ollama 서버 연결 및 기본 모델 테스트"""

    # Ollama 서버 URL
    base_url = os.getenv("OLLAMA_BASE_URL", "http://100.95.220.92:11434")

    print(f"🔌 Ollama 서버 연결 테스트: {base_url}")
    print("=" * 60)

    try:
        # 클라이언트 생성
        client = Client(host=base_url)

        # 1. 서버 상태 확인
        print("\n1. 서버 상태 확인...")
        models = client.list()

        if models.get('models'):
            print(f"✅ 서버 연결 성공! 설치된 모델 {len(models['models'])}개:")
            for model in models['models']:
                name = model.get('name', model.get('model', 'unknown'))
                size_gb = model.get('size', 0) / (1024**3)
                print(f"   • {name} ({size_gb:.2f}GB)")
        else:
            print("⚠️  서버는 연결되었지만 설치된 모델이 없습니다.")

        # 2. 간단한 모델 테스트 (가장 작은 모델로)
        test_model = "gemma2:2b"  # 작은 모델로 테스트
        print(f"\n2. 테스트 모델 다운로드 및 실행: {test_model}")

        # 모델이 없으면 다운로드
        model_names = [m['name'] for m in models.get('models', [])]
        if not any(test_model in name for name in model_names):
            print(f"   모델 다운로드 중... (첫 실행시 시간이 걸립니다)")
            client.pull(test_model)
            print(f"   ✅ 모델 다운로드 완료")

        # 3. 간단한 쿼리 테스트
        print(f"\n3. 쿼리 테스트...")
        response = client.generate(
            model=test_model,
            prompt="한국어로 '안녕하세요'라고 답해주세요.",
            options={
                "temperature": 0.1,
                "num_predict": 50,
            }
        )

        print(f"   질문: 한국어로 '안녕하세요'라고 답해주세요.")
        print(f"   답변: {response['response']}")
        print(f"   생성 시간: {response.get('total_duration', 0) / 1e9:.2f}초")

        print("\n✅ 모든 테스트 통과! Ollama 서버가 정상 작동 중입니다.")

        # 4. 권장 모델 목록
        print("\n📋 논문 실험용 권장 모델:")
        recommended = [
            ("gemma2:9b", "Google Gemma 2 (9B)"),
            ("qwen2.5:7b", "Alibaba Qwen 2.5 (7B)"),
            ("mistral:7b", "Mistral AI (7B)"),
            ("phi3.5:3.8b", "Microsoft Phi-3.5 (3.8B)"),
        ]

        for model_name, description in recommended:
            installed = "✅" if any(model_name in m for m in model_names) else "❌"
            print(f"   {installed} {model_name}: {description}")

        print("\n💡 모델 설치: ollama pull [모델명]")
        print("💡 예시: ollama pull gemma2:9b")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("\n문제 해결 방법:")
        print("1. Ollama 서버가 실행 중인지 확인")
        print("2. 방화벽 설정 확인")
        print("3. .env 파일의 OLLAMA_BASE_URL 확인")
        return False

    return True

if __name__ == "__main__":
    test_ollama_connection()