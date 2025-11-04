#!/usr/bin/env python3
"""Test GPT-OSS-20B model directly to diagnose empty response issue"""

import requests
import json

def test_gpt_oss():
    """Test GPT-OSS-20B with different parameters"""

    base_url = "http://100.95.220.92:11434"

    # Test 1: Simple generate
    print("Test 1: Simple generate request...")
    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": "gpt-oss:20b",
            "prompt": "Hello, please respond with any text.",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 100,
                "stop": []
            }
        },
        timeout=30
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response text: '{result.get('response', '')}'")
        print(f"Total duration: {result.get('total_duration', 0)/1e9:.2f}s")
        print(f"Model: {result.get('model', 'unknown')}")
    else:
        print(f"Error: {response.text}")

    print("\n" + "="*50 + "\n")

    # Test 2: Check model info
    print("Test 2: Checking model info...")
    response = requests.post(
        f"{base_url}/api/show",
        json={"name": "gpt-oss:20b"},
        timeout=10
    )

    if response.status_code == 200:
        info = response.json()
        print(f"Model family: {info.get('details', {}).get('family', 'unknown')}")
        print(f"Parameters: {info.get('details', {}).get('parameter_size', 'unknown')}")
        print(f"Quantization: {info.get('details', {}).get('quantization_level', 'unknown')}")

        # Check template
        template = info.get('template', '')
        print(f"Template found: {'Yes' if template else 'No'}")
        if template:
            print(f"Template preview: {template[:200]}...")

        # Check parameters
        params = info.get('parameters', '')
        if params:
            print(f"Model parameters: {params[:200]}...")
    else:
        print(f"Error getting model info: {response.text}")

    print("\n" + "="*50 + "\n")

    # Test 3: Try with different prompting style
    print("Test 3: Testing with different prompt formats...")

    test_prompts = [
        "Q: What is 2+2?\nA:",
        "User: Hello\nAssistant:",
        "<|user|>\nHello\n<|assistant|>",
        "### Instruction:\nSay hello\n\n### Response:",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"Format {i}: {prompt[:30]}...")
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": "gpt-oss:20b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "num_predict": 50,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            text = result.get('response', '').strip()
            if text:
                print(f"  Response: '{text[:100]}...'")
            else:
                print(f"  Response: <empty>")
        else:
            print(f"  Error: {response.status_code}")

    print("\n" + "="*50 + "\n")

    # Test 4: Check if model needs specific system prompt
    print("Test 4: Testing with system prompt...")
    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": "gpt-oss:20b",
            "prompt": "You are a helpful assistant. User: What is Python? Assistant:",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 150,
                "seed": 42
            }
        },
        timeout=30
    )

    if response.status_code == 200:
        result = response.json()
        text = result.get('response', '').strip()
        print(f"Response with system prompt: '{text[:200]}...' (length: {len(text)})")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    print("Testing GPT-OSS-20B model for empty response issue...")
    print("="*50)
    test_gpt_oss()