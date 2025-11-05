#!/bin/bash
# Test Claude as RAGAS judge model

set -e

echo "🔍 Testing Claude Code server connection..."

# Check server availability
if ! curl -s http://localhost:25292/v1/models > /dev/null 2>&1; then
    echo "❌ Claude Code server not responding on port 25292"
    echo "   Please start the server first"
    exit 1
fi

echo "✅ Server is running"

# Set environment variables
export CLAUDE_CODE_API_BASE="http://localhost:25292/v1"
export CLAUDE_CODE_API_KEY="dummy"

echo ""
echo "🚀 Running benchmark with Claude Sonnet as judge..."
echo "   Target model: gpt-4o-mini"
echo "   Questions: 5 (quick test)"
echo "   Judge: claude-sonnet-4-5-20250929"
echo ""

# Run benchmark
uv run python src/rag_pipeline/unified_benchmark_v2.py \
  --questions 5 \
  --models gpt-4o-mini \
  --judge-model claude-sonnet-4-5-20250929 \
  --force-generate

echo ""
echo "✅ Test completed!"
