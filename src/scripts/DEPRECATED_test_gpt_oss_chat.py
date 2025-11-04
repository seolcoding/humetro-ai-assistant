#!/usr/bin/env python3
"""Test GPT-OSS-20B using Ollama Chat API with proper Harmony format"""

import requests
import json

def test_gpt_oss_chat():
    base_url = "http://100.95.220.92:11434"

    # Test using the chat API which should handle Harmony format automatically
    print("Testing GPT-OSS-20B with Chat API (Harmony format handled by Ollama)...")
    print("="*60)

    test_messages = [
        {
            "prompt": "서울시 120 다산콜센터는 무엇인가요?",
            "category": "정보"
        },
        {
            "prompt": "What is Python programming language?",
            "category": "영어"
        },
        {
            "prompt": "안녕하세요, 오늘 날씨가 좋네요.",
            "category": "대화"
        }
    ]

    for test in test_messages:
        print(f"\n테스트: [{test['category']}] {test['prompt'][:30]}...")
        print("-"*40)

        # Using chat API - Ollama should handle Harmony format
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": "gpt-oss:20b",
                "messages": [
                    {"role": "user", "content": test['prompt']}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 256,
                    "seed": 42
                }
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            message = result.get('message', {})
            content = message.get('content', '')

            print(f"Role: {message.get('role', 'unknown')}")
            print(f"Response length: {len(content)} chars")

            if content:
                print(f"Response preview: {content[:200]}...")
            else:
                print("Response: <EMPTY>")

            # Check eval metrics
            eval_count = result.get('eval_count', 0)
            eval_duration = result.get('eval_duration', 0)
            if eval_duration > 0:
                tps = eval_count / (eval_duration / 1e9)
                print(f"Tokens/sec: {tps:.1f}")
        else:
            print(f"Error: HTTP {response.status_code}")
            print(response.text[:500])

    print("\n" + "="*60)
    print("결론:")
    print("- GPT-OSS-20B는 Harmony format을 사용합니다")
    print("- Ollama chat API가 자동으로 포맷을 처리해야 합니다")
    print("- 빈 응답이 나오면 모델 파일이나 템플릿 문제일 수 있습니다")

if __name__ == "__main__":
    test_gpt_oss_chat()