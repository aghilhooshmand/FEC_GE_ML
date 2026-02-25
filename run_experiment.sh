#!/bin/bash
# Batch script to install dependencies and run FEC_GE experiment

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== FEC_GE Experiment Runner ===${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found. Please install Python 3.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}Found: ${PYTHON_VERSION}${NC}"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip --quiet

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    pip install "numpy<2.0" pandas plotly mlflow deap scikit-learn scikit-learn-extra
fi

echo -e "${GREEN}✓ All dependencies installed${NC}"
echo ""

# Create results directory if it doesn't exist
mkdir -p results

# Run the main script (simple union vs baseline experiment)
echo -e "${YELLOW}Starting simple_union_vs_baseline experiment...${NC}"
echo ""
python3 simple_union_vs_baseline.py

echo ""
echo -e "${GREEN}=== Experiment Complete ===${NC}"
echo ""
echo "Results are saved in: ${SCRIPT_DIR}/results/"
echo ""

