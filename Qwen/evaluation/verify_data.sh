#!/bin/bash
# Verify Qwen3-VL evaluation datasets

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

DATA_DIR="${DATA_DIR:-./data}"

echo "=== Dataset Verification ==="
echo "Data directory: $DATA_DIR"
echo ""

verify_file() {
    local name=$1
    local file=$2
    local min_size_kb=$3
    local expected_cols=$4
    
    printf "%-20s " "$name:"
    
    if [ ! -f "$file" ]; then
        echo -e "${RED}✗ Not found${NC}"
        return 1
    fi
    
    size_kb=$(du -k "$file" | cut -f1)
    size_h=$(du -h "$file" | cut -f1)
    
    if [ "$size_kb" -lt "$min_size_kb" ]; then
        echo -e "${RED}✗ Too small ($size_h, expected >$((min_size_kb/1024))MB)${NC}"
        return 1
    fi
    
    # Check TSV structure
    if ! head -n 1 "$file" | grep -q $'\t'; then
        echo -e "${RED}✗ Invalid TSV format${NC}"
        return 1
    fi
    
    # Count columns
    col_count=$(head -n 1 "$file" | awk -F'\t' '{print NF}')
    
    # Count lines
    line_count=$(wc -l < "$file")
    
    if [ "$col_count" -ge "$expected_cols" ]; then
        echo -e "${GREEN}✓${NC} Valid ($size_h, $line_count lines, $col_count columns)"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} Valid but unexpected columns ($col_count, expected >=$expected_cols)"
        return 0
    fi
}

echo "Checking datasets..."
echo ""

all_valid=true

verify_file "MMMU" \
    "$DATA_DIR/MMMU/MMMU_DEV_VAL.tsv" \
    10000 \
    8 || all_valid=false

verify_file "MathVision" \
    "$DATA_DIR/MathVision/MathVision.tsv" \
    50000 \
    5 || all_valid=false

verify_file "RealWorldQA" \
    "$DATA_DIR/RealWorldQA/RealWorldQA.tsv" \
    30000 \
    7 || all_valid=false

echo ""
echo "ODinW-13: Auto-downloaded during inference (no pre-download needed)"

echo ""
if [ "$all_valid" = true ]; then
    echo -e "${GREEN}✓ All datasets verified successfully${NC}"
    exit 0
else
    echo -e "${RED}✗ Some datasets missing or invalid${NC}"
    echo "Run: bash setup_data.sh"
    exit 1
fi
