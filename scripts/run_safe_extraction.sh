#!/bin/bash
"""
Safe Knowledge Extraction Pipeline
====================================

1. Initialize cache if needed
2. Run extraction with 401 detection
3. Monitor progress
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
INPUT_FILE="data/processed/knowledge_extraction/full_dialogues.json"
CACHE_FILE=".cache/extraction_state.json"
CONFIG_FILE="config/knowledge_extraction_full.json"
LOG_DIR="logs"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🚀 SAFE KNOWLEDGE EXTRACTION PIPELINE${NC}"
echo -e "${GREEN}========================================${NC}"

# Step 1: Initialize cache if it doesn't exist
if [ ! -f "$CACHE_FILE" ]; then
    echo -e "${YELLOW}📋 Initializing cache...${NC}"
    uv run python scripts/initialize_extraction_cache.py \
        --input "$INPUT_FILE" \
        --cache "$CACHE_FILE"

    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Cache initialization failed${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Cache file exists: $CACHE_FILE${NC}"

    # Optionally verify cache
    echo -e "${YELLOW}🔍 Verifying cache...${NC}"
    uv run python scripts/initialize_extraction_cache.py \
        --input "$INPUT_FILE" \
        --cache "$CACHE_FILE" \
        --verify
fi

echo ""
echo -e "${YELLOW}📊 Starting extraction...${NC}"
echo -e "Input: $INPUT_FILE"
echo -e "Config: $CONFIG_FILE"
echo -e "Cache: $CACHE_FILE"
echo ""

# Step 2: Run extraction with monitoring
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting extraction in 3 seconds...${NC}"
echo -e "${GREEN}Open another terminal and run:${NC}"
echo -e "${YELLOW}uv run python scripts/monitor_extraction.py${NC}"
echo -e "${GREEN}========================================${NC}"
sleep 3

# Run extraction with debug mode
uv run python src/knowledge_extraction/claude_knowledge_extractor_v3.py \
    --config "$CONFIG_FILE" \
    --input "$INPUT_FILE" \
    --debug

EXIT_CODE=$?

# Check exit code
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Extraction completed successfully${NC}"
elif [ $EXIT_CODE -eq 401 ]; then
    echo -e "${RED}🔐 Authentication error (401) detected!${NC}"
    echo -e "${RED}Please check your API key and re-authenticate.${NC}"
    echo -e "${YELLOW}The extraction has been safely stopped.${NC}"
    echo -e "${YELLOW}You can resume by running this script again.${NC}"
else
    echo -e "${RED}❌ Extraction failed with code: $EXIT_CODE${NC}"
fi

# Show final statistics
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}📊 FINAL STATISTICS${NC}"
echo -e "${GREEN}========================================${NC}"

# Count cache status
if [ -f "$CACHE_FILE" ]; then
    SUCCESS_COUNT=$(grep -o '"status": "success"' "$CACHE_FILE" | wc -l)
    FAIL_COUNT=$(grep -o '"status": "fail"' "$CACHE_FILE" | wc -l)
    PENDING_COUNT=$(grep -o '"status": "pending"' "$CACHE_FILE" | wc -l)

    echo -e "✅ Success: $SUCCESS_COUNT"
    echo -e "❌ Failed: $FAIL_COUNT"
    echo -e "⏳ Pending: $PENDING_COUNT"
fi

echo -e "${GREEN}========================================${NC}"