#!/bin/bash
# Qwen3-VL Evaluation Data Setup Script

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default data directory
DATA_DIR="${DATA_DIR:-./data}"

echo -e "${GREEN}=== Qwen3-VL Evaluation Data Setup ===${NC}"
echo "Data directory: $DATA_DIR"
echo ""

# Create data directories
echo -e "${YELLOW}Step 1: Creating data directories...${NC}"
mkdir -p "$DATA_DIR"/{MMMU,MathVision,RealWorldQA,ODinW-13}
echo "✓ Created: $DATA_DIR/MMMU"
echo "✓ Created: $DATA_DIR/MathVision"
echo "✓ Created: $DATA_DIR/RealWorldQA"
echo "✓ Created: $DATA_DIR/ODinW-13"
echo ""

# Function to download dataset
download_dataset() {
    local name=$1
    local url=$2
    local output=$3
    local expected_min_size=$4  # in KB
    
    echo -e "${YELLOW}Downloading $name...${NC}"
    if [ -f "$output" ]; then
        size_kb=$(du -k "$output" | cut -f1)
        if [ "$size_kb" -gt "$expected_min_size" ]; then
            echo -e "${GREEN}✓${NC} $name already exists ($(du -h "$output" | cut -f1)), skipping"
            return 0
        else
            echo -e "${YELLOW}⚠${NC} Existing file too small, re-downloading..."
            rm "$output"
        fi
    fi
    
    # Try wget first
    if command -v wget &> /dev/null; then
        wget --show-progress --progress=bar:force -O "$output" "$url" 2>&1 || {
            echo -e "${RED}✗${NC} wget failed, trying curl..."
            rm -f "$output"
            if command -v curl &> /dev/null; then
                curl -L --progress-bar -o "$output" "$url" || {
                    echo -e "${RED}✗${NC} Failed to download $name"
                    return 1
                }
            else
                echo -e "${RED}✗${NC} Neither wget nor curl available"
                return 1
            fi
        }
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$output" "$url" || {
            echo -e "${RED}✗${NC} Failed to download $name"
            return 1
        }
    else
        echo -e "${RED}✗${NC} Neither wget nor curl available. Please install one of them."
        return 1
    fi
    
    # Verify download
    if [ -f "$output" ]; then
        size_kb=$(du -k "$output" | cut -f1)
        size_human=$(du -h "$output" | cut -f1)
        if [ "$size_kb" -gt "$expected_min_size" ]; then
            echo -e "${GREEN}✓${NC} Downloaded $name successfully ($size_human)"
            return 0
        else
            echo -e "${RED}✗${NC} Downloaded file too small ($size_human), may be corrupted"
            return 1
        fi
    else
        echo -e "${RED}✗${NC} Download failed"
        return 1
    fi
}

echo -e "${YELLOW}Step 2: Downloading datasets...${NC}"
echo "This may take several minutes depending on your connection."
echo ""

# Download MMMU dataset (~50MB)
download_dataset "MMMU" \
    "https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv" \
    "$DATA_DIR/MMMU/MMMU_DEV_VAL.tsv" \
    10000  # 10MB minimum

echo ""

# Download MathVision dataset (~200MB)
download_dataset "MathVision" \
    "https://opencompass.openxlab.space/utils/VLMEval/MathVision.tsv" \
    "$DATA_DIR/MathVision/MathVision.tsv" \
    50000  # 50MB minimum

echo ""

# Download RealWorldQA dataset (~150MB)
download_dataset "RealWorldQA" \
    "https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv" \
    "$DATA_DIR/RealWorldQA/RealWorldQA.tsv" \
    30000  # 30MB minimum

echo ""
echo -e "${YELLOW}Step 3: Verifying downloads...${NC}"

# Function to verify TSV file
verify_tsv() {
    local name=$1
    local file=$2
    
    if [ ! -f "$file" ]; then
        echo -e "${RED}✗${NC} $name not found"
        return 1
    fi
    
    # Check if it's a valid TSV (has tabs)
    if head -n 1 "$file" | grep -q $'\t'; then
        line_count=$(wc -l < "$file")
        echo -e "${GREEN}✓${NC} $name valid ($line_count lines)"
        return 0
    else
        echo -e "${RED}✗${NC} $name invalid (not a TSV file)"
        return 1
    fi
}

all_valid=true
verify_tsv "MMMU" "$DATA_DIR/MMMU/MMMU_DEV_VAL.tsv" || all_valid=false
verify_tsv "MathVision" "$DATA_DIR/MathVision/MathVision.tsv" || all_valid=false
verify_tsv "RealWorldQA" "$DATA_DIR/RealWorldQA/RealWorldQA.tsv" || all_valid=false

echo ""
if [ "$all_valid" = true ]; then
    echo -e "${GREEN}=== Data Setup Complete ===${NC}"
    echo -e "Data location: ${GREEN}$DATA_DIR${NC}"
    echo ""
    echo "Dataset summary:"
    echo "  • MMMU:        $(du -h "$DATA_DIR/MMMU/MMMU_DEV_VAL.tsv" | cut -f1) (~900 samples)"
    echo "  • MathVision:  $(du -h "$DATA_DIR/MathVision/MathVision.tsv" | cut -f1) (~3000 samples)"
    echo "  • RealWorldQA: $(du -h "$DATA_DIR/RealWorldQA/RealWorldQA.tsv" | cut -f1) (~700 samples)"
    echo ""
    echo "Next steps:"
    echo "1. Install Python dependencies:"
    echo "   cd mmmu && pip install -r requirements.txt"
    echo ""
    echo "2. Edit inference scripts (set model path and data dir):"
    echo "   nano mmmu/infer_instruct.sh"
    echo ""
    echo "3. Run inference (no judge model needed yet):"
    echo "   cd mmmu && bash infer_instruct.sh"
else
    echo -e "${RED}=== Data Setup Failed ===${NC}"
    echo "Some datasets failed to download or verify."
    echo "Please check the errors above and try again."
    echo ""
    echo "You can also download manually:"
    echo "  See SETUP_DATA.md for alternative download methods"
    exit 1
fi
