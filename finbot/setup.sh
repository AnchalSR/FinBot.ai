#!/bin/bash
# FinBot Quick Start Script for Linux/macOS

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         FinBot - Financial Advisor Chatbot Setup             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Create .env file
echo -e "\n${YELLOW}Creating environment configuration...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ Created .env file from template${NC}"
    echo -e "${YELLOW}⚠ Please edit .env file with your configuration${NC}"
else
    echo -e "${YELLOW}.env file already exists${NC}"
fi

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create necessary directories
echo -e "\n${YELLOW}Creating project directories...${NC}"
mkdir -p data/documents
mkdir -p embeddings
mkdir -p logs
mkdir -p checkpoints
echo -e "${GREEN}✓ Project directories created${NC}"

# Show next steps
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete! 🎉                         ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo ""
echo "1. Edit configuration (optional):"
echo "   nano .env"
echo ""
echo "2. Add financial documents:"
echo "   # Add PDF or TXT files to: data/documents/"
echo ""
echo "3. Start the backend API (Terminal 1):"
echo "   source venv/bin/activate"
echo "   python -m backend.api"
echo ""
echo "4. Start the frontend (Terminal 2):"
echo "   source venv/bin/activate"
echo "   streamlit run frontend/app.py"
echo ""
echo "5. Open in browser:"
echo "   http://localhost:8501"
echo ""
echo -e "${YELLOW}For Docker deployment:${NC}"
echo "   docker-compose up -d"
echo ""
echo "For detailed documentation, see:"
echo "   - README.md (Project overview)"
echo "   - deploy.md (Deployment guide)"
echo "   - SETUP_GUIDE.md (Configuration options)"
echo ""
