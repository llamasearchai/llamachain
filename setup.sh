#!/bin/bash

# LlamaChain Setup Script
# This script helps set up the LlamaChain development environment

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== LlamaChain Setup Script ===${NC}"
echo "This script will help you set up the LlamaChain development environment."

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1)
if [[ $python_version == *"Python 3.10"* ]] || [[ $python_version == *"Python 3.11"* ]]; then
    echo -e "${GREEN}Python version is compatible: $python_version${NC}"
else
    echo -e "${RED}Python 3.10+ is required, but found: $python_version${NC}"
    echo "Please install Python 3.10 or higher and try again."
    exit 1
fi

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created.${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}Virtual environment activated.${NC}"

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}Dependencies installed.${NC}"

# Create .env file if it doesn't exist
echo -e "\n${YELLOW}Setting up environment configuration...${NC}"
if [ -f ".env" ]; then
    echo ".env file already exists. Skipping creation."
else
    cp .env.example .env
    echo -e "${GREEN}.env file created from example. Please edit it with your configuration.${NC}"
fi

# Create necessary directories
echo -e "\n${YELLOW}Creating necessary directories...${NC}"
mkdir -p data/logs
mkdir -p data/db
mkdir -p data/models
echo -e "${GREEN}Directories created.${NC}"

# Install pre-commit hooks
echo -e "\n${YELLOW}Setting up pre-commit hooks...${NC}"
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo -e "${GREEN}Pre-commit hooks installed.${NC}"
else
    echo -e "${YELLOW}pre-commit not found. Skipping pre-commit hooks installation.${NC}"
    echo "You can install it with: pip install pre-commit"
fi

# Check for Docker
echo -e "\n${YELLOW}Checking for Docker...${NC}"
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}Docker and Docker Compose are installed.${NC}"
    echo "You can start the services with: docker-compose up -d"
else
    echo -e "${YELLOW}Docker and/or Docker Compose not found.${NC}"
    echo "For the full experience, please install Docker and Docker Compose."
    echo "You can still run the application without Docker, but you'll need to set up PostgreSQL and Redis manually."
fi

# Final instructions
echo -e "\n${GREEN}=== Setup Complete ===${NC}"
echo -e "To start the LlamaChain API server:"
echo -e "  ${YELLOW}python -m llamachain${NC}"
echo -e "\nTo start the worker processes:"
echo -e "  ${YELLOW}python -m llamachain --mode worker${NC}"
echo -e "\nTo use the CLI:"
echo -e "  ${YELLOW}python -m llamachain --mode cli --help${NC}"
echo -e "\nMake sure to edit the .env file with your configuration before starting the services."
echo -e "\n${GREEN}Happy coding with LlamaChain!${NC}" 