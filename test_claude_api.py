#!/usr/bin/env python3
"""
Quick test for Claude Code API endpoint via LangChain ChatOpenAI
"""

from langchain_openai import ChatOpenAI

# Test Claude Code endpoint
print("🔍 Testing Claude Code API endpoint...")
print(f"   Endpoint: http://localhost:25292/v1")
print(f"   Model: claude-sonnet-4-5-20250929")
print()

try:
    # Initialize ChatOpenAI with Claude endpoint
    llm = ChatOpenAI(
        model="claude-sonnet-4-5-20250929",
        openai_api_base="http://localhost:25292/v1",
        openai_api_key="dummy",
        temperature=1.0,
        max_tokens=100
    )

    # Simple test prompt
    response = llm.invoke("Say 'Hello from Claude Code API!' in Korean.")

    print("✅ API call successful!")
    print(f"Response: {response.content}")
    print()

except Exception as e:
    print(f"❌ API call failed: {e}")
    print()
    import traceback
    traceback.print_exc()
