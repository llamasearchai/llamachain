#!/bin/bash

# ========================================================
# LlamaChain: Blockchain Intelligence Platform
# A comprehensive tool for on-chain data analysis and
# smart contract security auditing
# ========================================================

# ANSI Color codes for llama-themed colorful CLI
PURPLE='\033[0;35m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

# Llama ASCII art
function display_llama_banner() {
  cat << "EOF"
                          ${PURPLE}   ██╗      ██╗      █████╗ ███╗   ███╗ █████╗  █████╗ ██╗  ██╗ █████╗ ██╗███╗   ██╗${NC}
                          ${PURPLE}   ██║      ██║     ██╔══██╗████╗ ████║██╔══██╗██╔══██╗██║  ██║██╔══██╗██║████╗  ██║${NC}
                          ${PURPLE}   ██║      ██║     ███████║██╔████╔██║███████║██║  ╚═╝███████║███████║██║██╔██╗ ██║${NC}
                          ${PURPLE}   ██║      ██║     ██╔══██║██║╚██╔╝██║██╔══██║██║  ██╗██╔══██║██╔══██║██║██║╚██╗██║${NC}
                          ${PURPLE}   ███████╗ ███████╗██║  ██║██║ ╚═╝ ██║██║  ██║╚█████╔╝██║  ██║██║  ██║██║██║ ╚████║${NC}
                          ${PURPLE}   ╚══════╝ ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝${NC}
                          
                                          ${YELLOW}Blockchain Analytics & Security Platform${NC}
EOF
}

function print_header() {
  echo -e "${YELLOW}====================================================================${NC}"
  echo -e "${YELLOW}    $1${NC}"
  echo -e "${YELLOW}====================================================================${NC}"
}

function log_info() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

function log_warning() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

function log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

function log_debug() {
  echo -e "${GRAY}[DEBUG]${NC} $1"
}

function log_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

function check_command() {
  if ! command -v $1 &> /dev/null; then
    log_error "$1 is not installed. Please install it to continue."
    return 1
  fi
  return 0
}

function check_prerequisites() {
  print_header "Checking Prerequisites"
  
  local failed=0
  
  # Check for Python 3.9+
  if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is not installed"
    failed=1
  else
    local python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if (( $(echo "$python_version < 3.9" | bc -l) )); then
      log_error "Python 3.9+ is required (found $python_version)"
      failed=1
    else
      log_info "Python $python_version detected"
    fi
  fi
  
  # Check for pip
  if ! command -v pip3 &> /dev/null; then
    log_error "pip3 is not installed"
    failed=1
  else
    log_info "pip3 detected"
  fi
  
  # Check for Docker
  if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed"
    failed=1
  else
    log_info "Docker detected"
  fi
  
  # Check for Docker Compose
  if ! command -v docker-compose &> /dev/null; then
    log_error "Docker Compose is not installed"
    failed=1
  else
    log_info "Docker Compose detected"
  fi
  
  # Check for git
  if ! command -v git &> /dev/null; then
    log_error "git is not installed"
    failed=1
  else
    log_info "git detected"
  fi
  
  # Check if this is running on macOS with Apple Silicon
  if [[ "$(uname)" == "Darwin" ]]; then
    log_info "macOS detected"
    
    # Check for Apple Silicon
    if [[ "$(uname -m)" == "arm64" ]]; then
      log_info "Apple Silicon (arm64) detected - MLX optimizations will be enabled"
      export LLAMACHAIN_ENABLE_MLX=1
    else
      log_warning "Intel Mac detected - MLX optimizations will not be available"
      export LLAMACHAIN_ENABLE_MLX=0
    fi
  else
    log_warning "Non-macOS system detected - MLX optimizations will not be available"
    export LLAMACHAIN_ENABLE_MLX=0
  fi
  
  # Check for brew on macOS
  if [[ "$(uname)" == "Darwin" ]]; then
    if ! command -v brew &> /dev/null; then
      log_error "Homebrew is not installed. It's recommended for macOS dependencies."
      failed=1
    else
      log_info "Homebrew detected"
    fi
  fi
  
  # Check for kubectl (optional, will be installed if missing)
  if ! command -v kubectl &> /dev/null; then
    log_warning "kubectl is not installed (will be installed later)"
  else
    log_info "kubectl detected"
  fi
  
  if [[ $failed -eq 1 ]]; then
    log_error "Please install the missing prerequisites and try again."
    return 1
  fi
  
  log_success "All core prerequisites are satisfied!"
  return 0
}

function create_virtual_environment() {
  print_header "Setting up Python Virtual Environment"
  
  # Create a virtual environment if it doesn't exist
  if [[ ! -d "venv" ]]; then
    log_info "Creating a new virtual environment..."
    python3 -m venv venv
    if [[ $? -ne 0 ]]; then
      log_error "Failed to create virtual environment"
      return 1
    fi
    log_success "Virtual environment created"
  else
    log_info "Using existing virtual environment"
  fi
  
  # Activate the virtual environment
  source venv/bin/activate
  if [[ $? -ne 0 ]]; then
    log_error "Failed to activate virtual environment"
    return 1
  fi
  log_success "Virtual environment activated"
  
  # Upgrade pip
  log_info "Upgrading pip..."
  pip install --upgrade pip
  
  return 0
}

function install_dependencies() {
  print_header "Installing Dependencies"
  
  # Make sure we're in the virtual environment
  if [[ "$VIRTUAL_ENV" == "" ]]; then
    log_error "Virtual environment not activated"
    return 1
  fi
  
  # Install dependencies
  log_info "Installing Python dependencies (this may take a while)..."
  pip install -r requirements.txt
  
  if [[ $? -ne 0 ]]; then
    log_error "Failed to install Python dependencies"
    return 1
  fi
  
  # Install system dependencies for macOS
  if [[ "$(uname)" == "Darwin" ]]; then
    log_info "Installing system dependencies with Homebrew..."
    
    # Check if Homebrew is installed
    if command -v brew &> /dev/null; then
      # Install GraphViz for visualization
      brew install graphviz
      
      # Install kubectl if not present
      if ! command -v kubectl &> /dev/null; then
        brew install kubectl
      fi
      
      # Install Go for Geth and other blockchain tools
      brew install go
      
      # Install Node.js and npm for frontend development
      brew install node
    else
      log_warning "Homebrew not installed, skipping system dependencies"
    fi
  fi
  
  # Install solc compiler
  log_info "Installing solc compiler..."
  python -c "from solcx import install_solc; install_solc(version='0.8.17')"
  
  log_success "Dependencies installed successfully"
  return 0
}

function setup_nlp_environment() {
  print_header "Setting up NLP Environment"
  
  # Make sure we're in the virtual environment
  if [[ "$VIRTUAL_ENV" == "" ]]; then
    log_error "Virtual environment not activated"
    return 1
  fi
  
  # Install spaCy language models
  log_info "Setting up spaCy models..."
  python scripts/install_spacy_models.py
  
  if [[ $? -ne 0 ]]; then
    log_warning "Failed to install spaCy models. Some NLP features may be limited."
  else
    log_success "NLP environment setup completed"
  fi
  
  return 0
}

function setup_environment() {
  # Check prerequisites
  check_prerequisites
  if [[ $? -ne 0 ]]; then
    return 1
  fi
  
  # Create virtual environment
  create_virtual_environment
  if [[ $? -ne 0 ]]; then
    return 1
  fi
  
  # Install dependencies
  install_dependencies
  if [[ $? -ne 0 ]]; then
    return 1
  fi
  
  # Setup NLP environment
  setup_nlp_environment
  if [[ $? -ne 0 ]]; then
    log_warning "NLP environment setup had issues, but continuing..."
  fi
  
  log_success "Environment setup completed successfully!"
  return 0
}

function start_services() {
  print_header "Starting LlamaChain Services"
  
  # Make sure we're in the virtual environment
  if [[ "$VIRTUAL_ENV" == "" ]]; then
    # Try to activate virtual environment
    if [[ -d "venv" ]]; then
      source venv/bin/activate
      if [[ $? -ne 0 ]]; then
        log_error "Failed to activate virtual environment"
        return 1
      fi
      log_info "Virtual environment activated"
    else
      log_error "Virtual environment not found. Please run setup first."
      return 1
    fi
  fi
  
  # Start services using Docker Compose
  log_info "Starting services with Docker Compose..."
  docker-compose up -d
  
  if [[ $? -ne 0 ]]; then
    log_error "Failed to start services with Docker Compose"
    return 1
  fi
  
  log_success "LlamaChain services started successfully!"
  log_info "API is available at: http://localhost:5000/docs"
  return 0
}

function stop_services() {
  print_header "Stopping LlamaChain Services"
  
  # Stop services using Docker Compose
  log_info "Stopping services with Docker Compose..."
  docker-compose down
  
  if [[ $? -ne 0 ]]; then
    log_error "Failed to stop services with Docker Compose"
    return 1
  fi
  
  log_success "LlamaChain services stopped successfully!"
  return 0
}

function show_help() {
  print_header "LlamaChain Help"
  
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --setup       Setup the environment and install dependencies"
  echo "  --start       Start LlamaChain services"
  echo "  --stop        Stop LlamaChain services"
  echo "  --restart     Restart LlamaChain services"
  echo "  --status      Show status of LlamaChain services"
  echo "  --help        Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0 --setup    # Setup the environment"
  echo "  $0 --start    # Start services"
  echo "  $0 --stop     # Stop services"
  echo ""
}

function show_status() {
  print_header "LlamaChain Services Status"
  
  # Show running containers
  docker-compose ps
  
  return 0
}

# Main entry point
function main() {
  display_llama_banner
  
  # Check if no arguments were provided
  if [[ $# -eq 0 ]]; then
    show_help
    exit 0
  fi
  
  # Parse command-line arguments
  case "$1" in
    --setup)
      setup_environment
      ;;
    --start)
      start_services
      ;;
    --stop)
      stop_services
      ;;
    --restart)
      stop_services && start_services
      ;;
    --status)
      show_status
      ;;
    --help)
      show_help
      ;;
    *)
      log_error "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
  
  exit $?
}

# Call main function with all arguments
main "$@"
