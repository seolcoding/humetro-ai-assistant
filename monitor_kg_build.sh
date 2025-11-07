#!/bin/bash
################################################################################
# Monitor KG Build Progress
#
# Shows real-time status of the KG building process
################################################################################

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

clear
echo "================================================================================"
echo "Knowledge Graph Build Monitor"
echo "================================================================================"
echo ""

# Check if build is running
PID_FILE="kg_build.pid"
if [ ! -f "$PID_FILE" ]; then
    echo -e "${RED}❌ No build process found${NC}"
    echo ""
    echo "Start build with:"
    echo "  ./run_full_kg_build.sh"
    exit 1
fi

PID=$(cat "$PID_FILE")
if ! ps -p $PID > /dev/null 2>&1; then
    echo -e "${RED}❌ Build process not running (stale PID: $PID)${NC}"
    rm "$PID_FILE"
    exit 1
fi

# Find most recent log file
LOG_FILE=$(ls -t kg_build_full_*.log 2>/dev/null | head -1)
if [ -z "$LOG_FILE" ]; then
    LOG_FILE="kg_build_full.log"
fi

echo -e "${GREEN}✅ Build running (PID: $PID)${NC}"
echo "   Log file: $LOG_FILE"
echo ""

# Extract progress information
STAGE=$(tail -100 "$LOG_FILE" | grep -E "\[Stage [0-9]/[0-9]\]" | tail -1)
FILES_PROCESSED=$(tail -100 "$LOG_FILE" | grep -oE "Processed: [0-9]+" | tail -1 | grep -oE "[0-9]+")
FILES_TOTAL=$(tail -100 "$LOG_FILE" | grep -oE "Total files: [0-9]+" | tail -1 | grep -oE "[0-9]+")

if [ -n "$STAGE" ]; then
    echo -e "${BLUE}Current Stage:${NC}"
    echo "  $STAGE"
    echo ""
fi

if [ -n "$FILES_PROCESSED" ] && [ -n "$FILES_TOTAL" ]; then
    PERCENT=$((FILES_PROCESSED * 100 / FILES_TOTAL))
    echo -e "${BLUE}Progress:${NC}"
    echo "  Files: $FILES_PROCESSED / $FILES_TOTAL ($PERCENT%)"

    # Estimate remaining time (rough estimate: 20 seconds per file)
    REMAINING=$((FILES_TOTAL - FILES_PROCESSED))
    TIME_REMAINING=$((REMAINING * 20))
    HOURS=$((TIME_REMAINING / 3600))
    MINUTES=$(((TIME_REMAINING % 3600) / 60))
    echo "  Estimated remaining: ${HOURS}h ${MINUTES}m"
    echo ""
fi

# Show recent errors if any
ERRORS=$(tail -200 "$LOG_FILE" | grep -i "error\|fail" | wc -l | tr -d ' ')
if [ "$ERRORS" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Recent errors: $ERRORS${NC}"
    echo "   Check log: tail -f $LOG_FILE | grep -i error"
    echo ""
fi

# Quick Neo4j stats
echo -e "${BLUE}Current Graph Stats:${NC}"
uv run python -c "
from src.kg_agent.utils.neo4j_for_adk import graphdb
try:
    result = graphdb.send_query('MATCH (n) RETURN count(n) as nodes')
    nodes = result['query_result'][0]['nodes']
    result = graphdb.send_query('MATCH ()-[r]->() RETURN count(r) as rels')
    rels = result['query_result'][0]['rels']
    result = graphdb.send_query('MATCH (c:Chunk) RETURN count(c) as chunks')
    chunks = result['query_result'][0]['chunks']
    print(f'  Nodes: {nodes:,}')
    print(f'  Relationships: {rels:,}')
    print(f'  Chunks: {chunks:,}')
    graphdb.close()
except Exception as e:
    print(f'  Error getting stats: {e}')
" 2>/dev/null

echo ""
echo "================================================================================"
echo "Commands"
echo "================================================================================"
echo ""
echo "Watch log:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Stop build:"
echo "  kill $PID && rm $PID_FILE"
echo ""
echo "Refresh monitor:"
echo "  watch -n 10 ./monitor_kg_build.sh"
echo ""
