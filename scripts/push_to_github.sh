#!/bin/bash
# Push Brain-Inspired AI to GitHub
# 推送类脑人工智能到GitHub

set -e

echo "=============================================="
echo "Brain-Inspired AI - GitHub Push Script"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git is not installed${NC}"
    exit 1
fi

# Get GitHub repository URL
echo ""
echo "Enter your GitHub repository URL:"
echo "Example: https://github.com/username/brain-inspired-ai.git"
read -r REPO_URL

if [ -z "$REPO_URL" ]; then
    echo -e "${RED}Error: Repository URL is required${NC}"
    exit 1
fi

# Initialize git if not already
cd "$(dirname "$0")/.."

if [ ! -d ".git" ]; then
    echo -e "${YELLOW}Initializing git repository...${NC}"
    git init
    git branch -M main
else
    echo -e "${GREEN}Git repository already initialized${NC}"
fi

# Add all files
echo ""
echo -e "${YELLOW}Adding files to git...${NC}"
git add -A

# Commit
echo ""
echo -e "${YELLOW}Creating commit...${NC}"
git commit -m "Initial commit: Brain-Inspired AI v1.0.0

Features:
- High-refresh-rate streaming inference (60Hz+)
- Spiking Neural Networks with LIF and Adaptive LIF neurons
- Online STDP learning for continuous adaptation
- Hippocampal memory system with pattern completion
- Multimodal processing (text, vision, audio)
- Tool integration (Wikipedia, web search, calculator)
- FastAPI server with WebSocket support
- Streamlit web interface
- Comprehensive benchmarking suite

Models:
- Base: Qwen2.5-0.5B-Instruct
- Vision: CLIP ViT-B/32

Documentation:
- English and Chinese README
- API documentation
- Example notebooks
- Docker support

See README.md for usage instructions."

# Add remote
echo ""
echo -e "${YELLOW}Adding remote repository...${NC}"
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"

# Push
echo ""
echo -e "${YELLOW}Pushing to GitHub...${NC}"
git push -u origin main --force

echo ""
echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}Successfully pushed to GitHub!${NC}"
echo -e "${GREEN}==============================================${NC}"
echo ""
echo "Repository URL: $REPO_URL"
echo ""
echo "Next steps:"
echo "1. Visit your GitHub repository"
echo "2. Add repository description and topics"
echo "3. Enable GitHub Actions for CI/CD"
echo "4. Create a release for v1.0.0"
echo ""
