#!/bin/bash
################################################################################
# Full Knowledge Graph Build with GPT-4o-mini
#
# This script processes 874 filtered markdown files (2021-2025 + timeless docs)
#
# Estimated:
#   - Time: 4-6 hours
#   - Cost: ~$2.30 (GPT-4o-mini + embeddings)
#   - Output: ~2,600 chunks with entities and relationships
################################################################################

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "Full Knowledge Graph Build - GPT-4o-mini"
echo "================================================================================"

# Configuration
MODEL="gpt-4o-mini"
NUM_FILES=0  # 0 = all files (874 filtered)
LOG_FILE="kg_build_full_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="kg_build.pid"

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "${RED}❌ Error: KG build already running (PID: $PID)${NC}"
        echo "   Check progress: tail -f $LOG_FILE"
        exit 1
    else
        echo "Cleaning up stale PID file..."
        rm "$PID_FILE"
    fi
fi

# Display estimates
echo ""
echo "Configuration:"
echo "  Model:      $MODEL"
echo "  Files:      874 filtered files (2021-2025 + timeless)"
echo "  Source:     data/crawled/seoul_traffic/markdown_filtered"
echo "  Log file:   $LOG_FILE"
echo ""
echo "Estimates:"
echo "  Duration:   ~4-6 hours"
echo "  Cost:       ~\$2.30"
echo "  Chunks:     ~2,600 chunks"
echo "  Entities:   ~500-1,000 domain entities"
echo ""
echo -e "${YELLOW}⚠️  This is a long-running process. Consider running in tmux/screen.${NC}"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Pre-flight check
echo ""
echo "Running pre-flight check..."
if uv run python src/kg_agent/preflight_check.py 2>&1 | grep -q "ALL CHECKS PASSED"; then
    echo -e "${GREEN}✅ Pre-flight check passed${NC}"
else
    echo -e "${RED}❌ Pre-flight check failed${NC}"
    echo "   Run: uv run python src/kg_agent/preflight_check.py"
    exit 1
fi

# Start build
echo ""
echo "Starting KG build in background..."
echo "  PID file: $PID_FILE"
echo "  Log file: $LOG_FILE"
echo ""

# Run in background and save PID
nohup uv run python src/kg_agent/build_kg.py \
    --num-files $NUM_FILES \
    --model $MODEL \
    > "$LOG_FILE" 2>&1 &

BUILD_PID=$!
echo $BUILD_PID > "$PID_FILE"

echo -e "${GREEN}✅ KG build started (PID: $BUILD_PID)${NC}"
echo ""
echo "================================================================================"
echo "Monitoring Commands"
echo "================================================================================"
echo ""
echo "Watch progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check if running:"
echo "  ps -p $BUILD_PID"
echo ""
echo "Stop build:"
echo "  kill $BUILD_PID"
echo "  rm $PID_FILE"
echo ""
echo "Check graph (while running):"
echo "  uv run python src/kg_agent/check_graph.py"
echo ""
echo "================================================================================"
echo ""
echo -e "${YELLOW}Tip: Run this in a separate terminal to watch progress:${NC}"
echo "  watch -n 60 'tail -30 $LOG_FILE'"
echo ""
echo "================================================================================"

# Wait a bit and show initial output
sleep 5
echo ""
echo "Initial output:"
echo "--------------------------------------------------------------------------------"
tail -20 "$LOG_FILE"
echo "--------------------------------------------------------------------------------"
echo ""
echo -e "${GREEN}Build is running in background. Use 'tail -f $LOG_FILE' to monitor.${NC}"
echo ""
