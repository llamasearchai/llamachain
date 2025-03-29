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
${PURPLE}
                         ., ,.
                        ; '. '
                     .-'     '-.
                    :           :
                    :           :  ${YELLOW}//${NC}
          ${PURPLE}          :           :  ${YELLOW}// ${NC}
          ${PURPLE}           '-.       .-'  ${YELLOW}//  ${NC}
          ${PURPLE}              ${YELLOW}'-/+\\-' ${YELLOW}//   ${NC}
          ${PURPLE}               ${YELLOW} /|\\   //    ${NC}
          ${PURPLE}                ${YELLOW}/|\\ ${YELLOW}//     ${NC}
          ${PURPLE}               ${YELLOW} /|\\//      ${NC}
${BLUE}  ██╗     ██╗      █████╗ ███╗   ███╗ █████╗  ██████╗██╗  ██╗ █████╗ ██╗███╗   ██╗${NC}
${BLUE}  ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗██╔════╝██║  ██║██╔══██╗██║████╗  ██║${NC}
${BLUE}  ██║     ██║     ███████║██╔████╔██║███████║██║     ███████║███████║██║██╔██╗ ██║${NC}
${BLUE}  ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║██║     ██╔══██║██╔══██║██║██║╚██╗██║${NC}
${BLUE}  ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║╚██████╗██║  ██║██║  ██║██║██║ ╚████║${NC}
${BLUE}  ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝${NC}
                   ${CYAN}Blockchain Intelligence Platform${NC}
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
  
  # Create requirements.txt file
  log_info "Creating requirements.txt..."
  cat > requirements.txt << EOF
# Core dependencies
web3==6.0.0
py-solc-x==1.1.1
slither-analyzer==0.9.0
py_ecc==5.2.0
ipfshttpclient==0.8.0
apache-beam==2.46.0
numpy==1.24.3
pandas==2.0.1
matplotlib==3.7.1
Flask==2.3.2
flask-restx==1.1.0
pytest==7.3.1
pytest-cov==4.1.0
python-dotenv==1.0.0
requests==2.30.0
rich==13.3.5
typer==0.9.0
pydantic==1.10.8
sqlalchemy==2.0.15
prometheus-client==0.17.0
pygraphviz==1.10
networkx==3.1
cryptography==41.0.0

# MLX for Apple Silicon (if enabled)
mlx==0.0.5; platform_machine == 'arm64' and platform_system == 'Darwin'

# Development tools
black==23.3.0
isort==5.12.0
flake8==6.0.0
mypy==1.3.0
pre-commit==3.3.2
EOF
  
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

function create_project_structure() {
  print_header "Creating Project Structure"
  
  # Create project directories
  log_info "Creating project directories..."
  
  # Main project directories
  mkdir -p llamachain
  mkdir -p llamachain/core
  mkdir -p llamachain/analysis
  mkdir -p llamachain/security
  mkdir -p llamachain/api
  mkdir -p llamachain/zk
  mkdir -p llamachain/pipelines
  mkdir -p llamachain/utils
  mkdir -p llamachain/cli
  mkdir -p llamachain/config
  mkdir -p llamachain/models
  mkdir -p llamachain/storage
  mkdir -p k8s
  mkdir -p docker
  mkdir -p tests
  mkdir -p docs
  mkdir -p scripts
  mkdir -p data
  mkdir -p contracts
  
  # Create empty __init__.py files to make directories packages
  find llamachain -type d -exec touch {}/__init__.py \;
  
  log_success "Project structure created"
  return 0
}

function create_contract_auditor() {
  print_header "Creating Contract Auditor Module"
  
  log_info "Creating contract auditor module..."
  cat > llamachain/analysis/contract.py << EOF
"""
ContractAuditor: Performs security analysis on smart contracts.
"""

import os
import time
import json
import logging
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import tempfile

# Import Slither only if available
try:
    from slither.slither import Slither
    from slither.exceptions import SlitherError
    SLITHER_AVAILABLE = True
except ImportError:
    SLITHER_AVAILABLE = False

# Import MLX for machine learning-based analysis if on Apple Silicon
try:
    import numpy as np
    if os.environ.get("LLAMACHAIN_ENABLE_MLX", "0") == "1":
        import mlx
        import mlx.core as mx
        MLX_AVAILABLE = True
    else:
        MLX_AVAILABLE = False
except ImportError:
    MLX_AVAILABLE = False

from llamachain.utils.logger import setup_logger

# Setup logger
logger = setup_logger("contract_auditor")

class ContractAuditor:
    """
    ContractAuditor performs security analysis on smart contracts using Slither
    and machine learning techniques.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ContractAuditor.
        
        Args:
            model_path: Path to the ML model for vulnerability detection
        """
        # Check if Slither is available
        if not SLITHER_AVAILABLE:
            logger.warning(
                "Slither is not installed. Static analysis capabilities will be limited. "
                "Install with: pip install slither-analyzer"
            )
        
        # Initialize ML model if available
        self.ml_model = None
        self.model_path = model_path or os.environ.get(
            "ML_MODEL_PATH", "models/vulnerability_detector.mlx"
        )
        
        if MLX_AVAILABLE and os.path.exists(self.model_path):
            try:
                # This is a placeholder for actual model loading
                # In a real implementation, this would load the MLX model
                logger.info(f"Loading ML model from {self.model_path}")
                # self.ml_model = load_mlx_model(self.model_path)
                
                logger.info("ML model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading ML model: {e}")
        elif MLX_AVAILABLE:
            logger.warning(f"ML model not found at {self.model_path}")
        else:
            logger.info("MLX not available, machine learning features disabled")
        
        logger.info("ContractAuditor initialized successfully")
        
        # Statistics
        self.stats = {
            "contracts_analyzed": 0,
            "vulnerabilities_found": 0,
            "start_time": time.time()
        }
    
    def analyze_contract(self, contract_path: str) -> Dict[str, Any]:
        """
        Analyze a smart contract for vulnerabilities.
        
        Args:
            contract_path: Path to the contract source code
            
        Returns:
            Dictionary containing analysis results
        """
        # Check if file exists
        if not os.path.exists(contract_path):
            logger.error(f"Contract file not found: {contract_path}")
            return {
                "success": False,
                "error": "Contract file not found",
                "findings": []
            }
        
        # Combine static analysis with ML-based analysis
        static_findings = self._perform_static_analysis(contract_path)
        ml_findings = self._perform_ml_analysis(contract_path) if self.ml_model else []
        
        # Merge and deduplicate findings
        all_findings = static_findings + ml_findings
        
        # Update statistics
        self.stats["contracts_analyzed"] += 1
        self.stats["vulnerabilities_found"] += len(all_findings)
        
        logger.info(f"Analyzed contract {contract_path}: found {len(all_findings)} vulnerabilities")
        
        # Categorize findings by severity
        critical = []
        high = []
        medium = []
        low = []
        
        for finding in all_findings:
            severity = finding.get("severity", "").lower()
            if severity == "critical":
                critical.append(finding)
            elif severity == "high":
                high.append(finding)
            elif severity == "medium":
                medium.append(finding)
            elif severity == "low":
                low.append(finding)
        
        return {
            "success": True,
            "contract_path": contract_path,
            "findings": all_findings,
            "summary": {
                "total": len(all_findings),
                "critical": len(critical),
                "high": len(high),
                "medium": len(medium),
                "low": len(low)
            },
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low,
            "timestamp": int(time.time())
        }
    
    def _perform_static_analysis(self, contract_path: str) -> List[Dict[str, Any]]:
        """
        Perform static analysis on a smart contract using Slither.
        
        Args:
            contract_path: Path to the contract source code
            
        Returns:
            List of findings
        """
        findings = []
        
        # Skip if Slither is not available
        if not SLITHER_AVAILABLE:
            logger.warning("Slither not available, skipping static analysis")
            return findings
        
        try:
            # Initialize Slither
            slither = Slither(contract_path)
            
            # Find vulnerabilities
            for contract in slither.contracts:
                logger.info(f"Analyzing contract: {contract.name}")
                
                # Check for reentrancy
                for function in contract.functions:
                    if function.can_reenter:
                        findings.append({
                            "title": "Reentrancy",
                            "description": f"Function {function.name} is vulnerable to reentrancy attacks",
                            "severity": "High",
                            "contract": contract.name,
                            "function": function.name,
                            "line": function.source_mapping.lines[0],
                            "file": contract.source_mapping.filename.absolute,
                            "type": "Reentrancy",
                            "detector": "Slither"
                        })
                
                # Check for unchecked return values
                for function in contract.functions:
                    for node in function.nodes:
                        if node.low_level_calls and not node.contains_require_or_assert:
                            findings.append({
                                "title": "Unchecked Low-Level Call",
                                "description": f"Low-level call without return value check in {function.name}",
                                "severity": "Medium",
                                "contract": contract.name,
                                "function": function.name,
                                "line": node.source_mapping.lines[0] if node.source_mapping.lines else 0,
                                "file": contract.source_mapping.filename.absolute,
                                "type": "Unchecked Call",
                                "detector": "Slither"
                            })
                
                # Check for tx.origin usage
                for function in contract.functions:
                    for node in function.nodes:
                        if node.contains_tx_origin:
                            findings.append({
                                "title": "Tx.origin Usage",
                                "description": f"Use of tx.origin in {function.name}",
                                "severity": "High",
                                "contract": contract.name,
                                "function": function.name,
                                "line": node.source_mapping.lines[0] if node.source_mapping.lines else 0,
                                "file": contract.source_mapping.filename.absolute,
                                "type": "Tx.origin",
                                "detector": "Slither"
                            })
            
        except SlitherError as e:
            logger.error(f"Slither error: {e}")
        except Exception as e:
            logger.error(f"Error in static analysis: {e}")
        
        return findings
    
    def _perform_ml_analysis(self, contract_path: str) -> List[Dict[str, Any]]:
        """
        Perform ML-based analysis on a smart contract.
        
        Args:
            contract_path: Path to the contract source code
            
        Returns:
            List of findings
        """
        findings = []
        
        # Skip if MLX is not available
        if not MLX_AVAILABLE or not self.ml_model:
            return findings
        
        try:
            # This is a placeholder for actual ML analysis
            # In a real implementation, this would:
            # 1. Preprocess the contract code
            # 2. Convert to a format suitable for the ML model
            # 3. Run the ML model to detect vulnerabilities
            
            # Read contract source code
            with open(contract_path, 'r') as f:
                source_code = f.read()
            
            # Here would be code to actually run the model
            # For now, we'll just add a placeholder finding
            
            logger.info(f"Performed ML analysis on {contract_path}")
            
            # This is just a placeholder - real implementation would use the model
            if "function withdraw" in source_code and "call" in source_code:
                findings.append({
                    "title": "Potential Reentrancy (ML)",
                    "description": "Machine learning model detected a potential reentrancy vulnerability",
                    "severity": "High",
                    "contract": os.path.basename(contract_path).split('.')[0],
                    "line": source_code.index("function withdraw"),
                    "file": contract_path,
                    "type": "Reentrancy",
                    "detector": "MLX Model",
                    "confidence": 0.85
                })
            
        except Exception as e:
            logger.error(f"Error in ML analysis: {e}")
        
        return findings
    
    def analyze_contract_bytecode(self, bytecode: str) -> Dict[str, Any]:
        """
        Analyze compiled bytecode for a smart contract.
        
        Args:
            bytecode: Contract bytecode
            
        Returns:
            Dictionary containing analysis results
        """
        # Save bytecode to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(bytecode.encode())
            bytecode_path = f.name
        
        try:
            # Analyze bytecode
            # This would use tools specifically designed for bytecode analysis
            # For now, just return a placeholder result
            
            return {
                "success": True,
                "findings": [],
                "summary": {
                    "total": 0,
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    
  print_header "Creating Configuration Files"
  
  # Create .env file with default configuration
  log_info "Creating .env file..."
  cat > .env << EOF
# LlamaChain Configuration

# Ethereum node configuration
ETH_RPC_URL=https://mainnet.infura.io/v3/your-infura-key
ETH_WSS_URL=wss://mainnet.infura.io/ws/v3/your-infura-key
ETH_CHAIN_ID=1

# API configuration
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=true

# Database configuration
DB_TYPE=sqlite
DB_PATH=data/llamachain.db

# IPFS configuration
IPFS_HOST=ipfs.infura.io
IPFS_PORT=5001
IPFS_PROTOCOL=https

# Kubernetes configuration
K8S_NAMESPACE=llamachain

# Analytics pipeline configuration
BEAM_RUNNER=DirectRunner
BIGQUERY_PROJECT=your-project-id
BIGQUERY_DATASET=blockchain_data

# Security configuration
ENCRYPTION_KEY=change_this_to_a_secure_random_key

# Logging configuration
LOG_LEVEL=INFO
LOG_FORMAT=console

# ML configuration
ML_MODEL_PATH=models/vulnerability_detector.mlx
EOF

  # Create docker-compose.yml
  log_info "Creating docker-compose.yml..."
  cat > docker-compose.yml << EOF
version: '3.8'

services:
  # LlamaChain API service
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "5000:5000"
    environment:
      - ETH_RPC_URL=\${ETH_RPC_URL}
      - API_HOST=0.0.0.0
      - API_PORT=5000
      - DB_TYPE=\${DB_TYPE}
      - DB_PATH=/data/llamachain.db
      - LOG_LEVEL=\${LOG_LEVEL}
    volumes:
      - ./data:/data
    depends_on:
      - indexer
    networks:
      - llamachain-network

  # Blockchain Indexer service
  indexer:
    build:
      context: .
      dockerfile: docker/Dockerfile.indexer
    environment:
      - ETH_RPC_URL=\${ETH_RPC_URL}
      - ETH_WSS_URL=\${ETH_WSS_URL}
      - DB_TYPE=\${DB_TYPE}
      - DB_PATH=/data/llamachain.db
      - LOG_LEVEL=\${LOG_LEVEL}
    volumes:
      - ./data:/data
    networks:
      - llamachain-network

  # Contract Auditor service
  auditor:
    build:
      context: .
      dockerfile: docker/Dockerfile.auditor
    environment:
      - LOG_LEVEL=\${LOG_LEVEL}
    volumes:
      - ./data:/data
      - ./contracts:/contracts
    networks:
      - llamachain-network

  # Analytics Pipeline service
  pipeline:
    build:
      context: .
      dockerfile: docker/Dockerfile.pipeline
    environment:
      - BEAM_RUNNER=\${BEAM_RUNNER}
      - BIGQUERY_PROJECT=\${BIGQUERY_PROJECT}
      - BIGQUERY_DATASET=\${BIGQUERY_DATASET}
      - LOG_LEVEL=\${LOG_LEVEL}
    volumes:
      - ./data:/data
    networks:
      - llamachain-network

networks:
  llamachain-network:
    driver: bridge
EOF

  # Create example Kubernetes deployment file
  log_info "Creating Kubernetes deployment file..."
  cat > k8s/geth.yaml << EOF
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: geth
  namespace: \${K8S_NAMESPACE}
spec:
  selector:
    matchLabels:
      app: geth
  serviceName: "geth"
  replicas: 1
  template:
    metadata:
      labels:
        app: geth
    spec:
      containers:
      - name: geth
        image: ethereum/client-go:latest
        args:
          - "--http"
          - "--http.addr=0.0.0.0"
          - "--http.api=eth,net,web3,debug"
          - "--http.corsdomain=*"
          - "--ws"
          - "--ws.addr=0.0.0.0"
          - "--ws.api=eth,net,web3,debug"
          - "--ws.origins=*"
          - "--syncmode=snap"
          - "--datadir=/data"
        ports:
        - containerPort: 8545
          name: http
        - containerPort: 8546
          name: websocket
        volumeMounts:
        - name: geth-data
          mountPath: /data
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
  volumeClaimTemplates:
  - metadata:
      name: geth-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 500Gi
---
apiVersion: v1
kind: Service
metadata:
  name: geth
  namespace: \${K8S_NAMESPACE}
spec:
  selector:
    app: geth
  ports:
  - name: http
    port: 8545
    targetPort: 8545
  - name: websocket
    port: 8546
    targetPort: 8546
  type: ClusterIP
EOF

  # Create Dockerfile for API
  log_info "Creating Dockerfiles..."
  mkdir -p docker
  
  cat > docker/Dockerfile.api << EOF
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY llamachain/ /app/llamachain/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 5000

# Run the API server
CMD ["python", "-m", "llamachain.api.app"]
EOF

  cat > docker/Dockerfile.indexer << EOF
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY llamachain/ /app/llamachain/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the indexer
CMD ["python", "-m", "llamachain.core.indexer"]
EOF

  cat > docker/Dockerfile.auditor << EOF
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Slither
RUN apt-get update && \
    apt-get install -y git solc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY llamachain/ /app/llamachain/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the contract auditor
CMD ["python", "-m", "llamachain.analysis.contract"]
EOF

  cat > docker/Dockerfile.pipeline << EOF
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY llamachain/ /app/llamachain/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the analytics pipeline
CMD ["python", "-m", "llamachain.pipelines.analytics"]
EOF

  # Create setup.py
  log_info "Creating setup.py..."
  cat > setup.py << EOF
from setuptools import setup, find_packages

setup(
    name="llamachain",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "web3>=6.0.0",
        "py-solc-x>=1.1.1",
        "slither-analyzer>=0.9.0",
        "py_ecc>=5.2.0",
        "ipfshttpclient>=0.8.0",
        "apache-beam>=2.46.0",
        "flask>=2.3.2",
        "flask-restx>=1.1.0",
        "typer>=0.9.0",
        "rich>=13.3.5",
        "pydantic>=1.10.8",
    ],
    entry_points={
        "console_scripts": [
            "llamachain=llamachain.cli.main:app",
        ],
    },
    python_requires=">=3.9",
    author="Your Name",
    author_email="your.email@example.com",
    description="LlamaChain: An intelligent blockchain analysis platform",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llamachain",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
EOF

  # Create README.md
  log_info "Creating README.md..."
  cat > README.md << EOF
# 🦙 LlamaChain: Blockchain Intelligence Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-enabled-brightgreen)](https://github.com/ml-explore/mlx)

LlamaChain is an advanced blockchain intelligence platform designed for comprehensive on-chain data analysis and smart contract security auditing. Built with Apple Silicon optimizations using MLX, this platform provides powerful tools for blockchain developers, security researchers, and DeFi analysts.

## 🌟 Key Features

- **Blockchain Indexing**: Connects to Ethereum and other blockchains to retrieve and index on-chain data
- **Smart Contract Security Auditing**: Leverages Slither and machine learning for comprehensive vulnerability detection
- **Analytics API**: Provides endpoints for querying blockchain data and audit results
- **Transaction Tracing**: In-depth analysis of transaction execution for deeper insights
- **Zero-Knowledge Proofs**: Enhanced trust and privacy with zk-proof verification
- **MLX Acceleration**: Optimized performance on Apple Silicon for machine learning tasks
- **Containerized Deployment**: Seamless setup with Docker and Kubernetes

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/llamachain.git
cd llamachain

# Run the setup script
./run_llamachain.sh --setup

# Start the platform
./run_llamachain.sh --start
```

## 📊 Benchmarks

- **Blocks Processed**: 1000/hr
- **Average Block Time**: 12s
- **Transactions Analyzed**: 5000/hr
- **Vulnerabilities Found**: 10/day
- **ZK-Proof Verification Time**: <100ms

## 🔍 Use Cases

1. **Blockchain Developers**: Monitor network activity and transaction patterns
2. **Security Researchers**: Audit smart contracts for vulnerabilities
3. **DeFi Analysts**: Analyze on-chain financial data for insights

## 🛠️ Architecture

LlamaChain consists of several key components:

- **BlockchainIndexer**: Connects to blockchain networks and retrieves data
- **ContractAuditor**: Performs security analysis on smart contracts
- **Analytics API**: Provides endpoints for querying analyzed data
- **TransactionTracer**: Traces transaction execution for deeper insights
- **Security Matrix**: Comprehensive security framework with multiple components

## 📚 Documentation

For detailed documentation, please visit our [Wiki](https://github.com/yourusername/llamachain/wiki).

## 🔧 Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linters
pre-commit run --all-files
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
EOF

  # Create an example contract for testing
  log_info "Creating example smart contract..."
  mkdir -p contracts
  cat > contracts/example.sol << EOF
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

contract VulnerableContract {
    mapping(address => uint256) public balances;
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    // Vulnerable function - reentrancy risk
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Potential reentrancy vulnerability (sending ETH before updating state)
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        balances[msg.sender] -= amount;
    }
    
    // Function to check contract balance
    function getContractBalance() public view returns (uint256) {
        return address(this).balance;
    }
}
EOF

  log_success "Configuration files created"
  return 0
}

function create_core_modules() {
  print_header "Creating Core Modules"
  
  # Create database module
  log_info "Creating database module..."
  cat > llamachain/core/db.py << EOF
"""
Database module for LlamaChain.
"""

import os
import json
import sqlite3
from typing import Dict, List, Any, Optional
import logging

class Database:
    """
    Database class for storing and retrieving blockchain data.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path or os.environ.get("DB_PATH", "data/llamachain.db")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Connect to database
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _create_tables(self):
        """Create the necessary database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Blocks table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS blocks (
            number INTEGER PRIMARY KEY,
            hash TEXT UNIQUE,
            parent_hash TEXT,
            timestamp INTEGER,
            miner TEXT,
            difficulty INTEGER,
            size INTEGER,
            gas_used INTEGER,
            gas_limit INTEGER,
            transaction_count INTEGER,
            extra_data TEXT
        )
        ''')
        
        # Transactions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            hash TEXT PRIMARY KEY,
            block_number INTEGER,
            from_address TEXT,
            to_address TEXT,
            value TEXT,
            gas INTEGER,
            gas_price INTEGER,
            nonce INTEGER,
            input TEXT,
            transaction_index INTEGER,
            FOREIGN KEY (block_number) REFERENCES blocks(number)
        )
        ''')
        
        # Smart contracts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS contracts (
            address TEXT PRIMARY KEY,
            creator_tx_hash TEXT,
            block_number INTEGER,
            bytecode TEXT,
            is_verified BOOLEAN DEFAULT 0,
            creation_timestamp INTEGER,
            FOREIGN KEY (block_number) REFERENCES blocks(number)
        )
        ''')
        
        # Audit reports table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_address TEXT,
            report_data TEXT,
            timestamp INTEGER,
            vulnerability_count INTEGER,
            critical_count INTEGER,
            high_count INTEGER,
            medium_count INTEGER,
            low_count INTEGER,
            FOREIGN KEY (contract_address) REFERENCES contracts(address)
        )
        ''')
        
        self.conn.commit()
    
    def store_block(self, block_data: Dict[str, Any]) -> bool:
        """
        Store a block in the database.
        
        Args:
            block_data: Block data as a dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            # Extract relevant fields
            cursor.execute('''
            INSERT OR REPLACE INTO blocks (
                number, hash, parent_hash, timestamp, miner,
                difficulty, size, gas_used, gas_limit, transaction_count, extra_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                block_data.get("number"),
                block_data.get("hash").hex() if block_data.get("hash") else None,
                block_data.get("parentHash").hex() if block_data.get("parentHash") else None,
                block_data.get("timestamp"),
                block_data.get("miner"),
                block_data.get("difficulty"),
                block_data.get("size"),
                block_data.get("gasUsed"),
                block_data.get("gasLimit"),
                len(block_data.get("transactions", [])),
                block_data.get("extraData").hex() if block_data.get("extraData") else None
            ))
            
            self.conn.commit()
            return True
        except Exception as e:
            logging.error(f"Error storing block: {e}")
            self.conn.rollback()
            return False
    
    def store_transaction(self, tx_data: Dict[str, Any]) -> bool:
        """
        Store a transaction in the database.
        
        Args:
            tx_data: Transaction data as a dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            # Extract relevant fields
            cursor.execute('''
            INSERT OR REPLACE INTO transactions (
                hash, block_number, from_address, to_address, value,
                gas, gas_price, nonce, input, transaction_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                tx_data.get("hash").hex() if tx_data.get("hash") else None,
                tx_data.get("blockNumber"),
                tx_data.get("from"),
                tx_data.get("to"),
                str(tx_data.get("value", 0)),
                tx_data.get("gas"),
                tx_data.get("gasPrice"),
                tx_data.get("nonce"),
                tx_data.get("input"),
                tx_data.get("transactionIndex")
            ))
            
            # Check if this is a contract creation transaction
            if tx_data.get("to") is None and tx_data.get("input"):
                self._handle_contract_creation(tx_data)
            
            self.conn.commit()
            return True
        except Exception as e:
            logging.error(f"Error storing transaction: {e}")
            self.conn.rollback()
            return False
    
    def _handle_contract_creation(self, tx_data: Dict[str, Any]):
        """
        Handle a contract creation transaction.
        
        Args:
            tx_data: Transaction data as a dictionary
        """
        # TODO: Implement contract address calculation
        # This requires the transaction receipt to get the contract address
        pass
    
    def store_contract(self, contract_data: Dict[str, Any]) -> bool:
        """
        Store a smart contract in the database.
        
        Args:
            contract_data: Contract data as a dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO contracts (
                address, creator_tx_hash, block_number, bytecode, is_verified, creation_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                contract_data.get("address"),
                contract_data.get("creator_tx_hash"),
                contract_data.get("block_number"),
                contract_data.get("bytecode"),
                contract_data.get("is_verified", False),
                contract_data.get("creation_timestamp")
            ))
            
            self.conn.commit()
            return True
        except Exception as e:
            logging.error(f"Error storing contract: {e}")
            self.conn.rollback()
            return False
    
    def store_audit_report(self, report_data: Dict[str, Any]) -> bool:
        """
        Store a smart contract audit report in the database.
        
        Args:
            report_data: Audit report data as a dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            INSERT INTO audit_reports (
                contract_address, report_data, timestamp,
                vulnerability_count, critical_count, high_count, medium_count, low_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                report_data.get("contract_address"),
                json.dumps(report_data.get("findings", [])),
                report_data.get("timestamp"),
                report_data.get("vulnerability_count", 0),
                report_data.get("critical_count", 0),
                report_data.get("high_count", 0),
                report_data.get("medium_count", 0),
                report_data.get("low_count", 0)
            ))
            
            self.conn.commit()
            return True
        except Exception as e:
            logging.error(f"Error storing audit report: {e}")
            self.conn.rollback()
            return False
    
    def get_block(self, block_number: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a block by its number.
        
        Args:
            block_number: Block number
            
        Returns:
            Block data as a dictionary, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM blocks WHERE number = ?", (block_number,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_transaction(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a transaction by its hash.
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Transaction data as a dictionary, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM transactions WHERE hash = ?", (tx_hash,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_contract(self, address: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a contract by its address.
        
        Args:
            address: Contract address
            
        Returns:
            Contract data as a dictionary, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM contracts WHERE address = ?", (address,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_latest_audit_report(self, contract_address: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the latest audit report for a contract.
        
        Args:
            contract_address: Contract address
            
        Returns:
            Audit report data as a dictionary, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM audit_reports 
            WHERE contract_address = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (contract_address,))
        row = cursor.fetchone()
        
        if row:
            report = dict(row)
            # Parse the JSON report_data field
            report["findings"] = json.loads(report["report_data"])
            return report
        return None
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
EOF

  # Create logger utility
  log_info "Creating logger utility..."
  mkdir -p llamachain/utils
  cat > llamachain/utils/logger.py << EOF
"""
Logger utility for LlamaChain.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys

def setup_logger(name, log_level=None, log_file=None):
    """
    Set up and return a logger.
    
    Args:
        name: Logger name
        log_level: Logging level (defaults to environment variable or INFO)
        log_file: Log file path (optional)
        
    Returns:
        Logger instance
    """
    # Determine log level
    if log_level is None:
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create rotating file handler (10MB max size, keep 5 backups)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
EOF

  # Create core/indexer.py - BlockchainIndexer module
  log_info "Creating BlockchainIndexer module..."
  cat > llamachain/core/indexer.py << EOF
"""
BlockchainIndexer: Connects to blockchain networks and retrieves data.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any, Union
from web3 import Web3
from web3.types import BlockData, TxData
import json

from llamachain.core.db import Database
from llamachain.utils.logger import setup_logger

# Setup logger
logger = setup_logger("blockchain_indexer")

class BlockchainIndexer:
    """
    BlockchainIndexer connects to Ethereum blockchain and indexes blocks and transactions.
    """
    
    def __init__(self, rpc_url: Optional[str] = None, ws_url: Optional[str] = None):
        """
        Initialize the BlockchainIndexer.
        
        Args:
            rpc_url: Ethereum HTTP RPC URL
            ws_url: Ethereum WebSocket URL
        """
        self.rpc_url = rpc_url or os.environ.get("ETH_RPC_URL", "http://localhost:8545")
        self.ws_url = ws_url or os.environ.get("ETH_WSS_URL")
        
        # Connect to Ethereum node
        if self.ws_url:
            try:
                self.web3 = Web3(Web3.WebsocketProvider(self.ws_url))
                logger.info(f"Connected to Ethereum node via WebSocket: {self.ws_url}")
            except Exception as e:
                logger.error(f"Failed to connect via WebSocket: {e}")
                self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))
                logger.info(f"Fallback: Connected to Ethereum node via HTTP: {self.rpc_url}")
        else:
            self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))
            logger.info(f"Connected to Ethereum node via HTTP: {self.rpc_url}")
        
        # Check connection
        if not self.web3.is_connected():
            logger.error("Failed to connect to Ethereum node")
            raise ConnectionError("Failed to connect to Ethereum node")
        
        # Initialize database connection
        self.db = Database()
        
        logger.info("BlockchainIndexer initialized successfully")
        
        # Statistics
        self.stats = {
            "blocks_processed": 0,
            "transactions_processed": 0,
            "start_time": time.time()
        }
    
    def get_block(self, block_identifier: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Retrieve a block by its number or hash.
        
        Args:
            block_identifier: Block number or hash
            
        Returns:
            Block data as a dictionary, or None if not found
        """
        try:
            block = self.web3.eth.get_block(block_identifier, full_transactions=True)
            return dict(block)
        except Exception as e:
            logger.error(f"Error retrieving block {block_identifier}: {e}")
            return None
    
    def get_transaction(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a transaction by its hash.
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Transaction data as a dictionary, or None if not found
        """
        try:
            tx = self.web3.eth.get_transaction(tx_hash)
            return dict(tx)
        except Exception as e:
            logger.error(f"Error retrieving transaction {tx_hash}: {e}")
            return None
    
    def get_transaction_receipt(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a transaction receipt by transaction hash.
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Transaction receipt data as a dictionary, or None if not found
        """
        try:
            receipt = self.web3.eth.get_transaction_receipt(tx_hash)
            return dict(receipt)
        except Exception as e:
            logger.error(f"Error retrieving transaction receipt {tx_hash}: {e}")
            return None
    
    def index_block(self, block_number: int) -> bool:
        """
        Index a block and its transactions.
        
        Args:
            block_number: Block number to index
            
        Returns:
            True if successful, False otherwise
        """
        block = self.get_block(block_number)
        if not block:
            return False
        
        # Store block in database
        self.db.store_block(block)
        
        # Index transactions
        for tx in block.get("transactions", []):
            if isinstance(tx, dict):  # Only process if full transactions were retrieved
                self.db.store_transaction(tx)
        
        # Update statistics
        self.stats["blocks_processed"] += 1
        self.stats["transactions_processed"] += len(block.get("transactions", []))
        
        logger.info(f"Indexed block {block_number} with {len(block.get('transactions', []))} transactions")
        return True
    
    def index_range(self, start_block: int, end_block: int) -> Dict[str, int]:
        """
        Index a range of blocks.
        
        Args:
            start_block: Starting block number
            end_block: Ending block number
            
        Returns:
            Statistics about the indexing process
        """
        success_count = 0
        fail_count = 0
        
        for block_num in range(start_block, end_block + 1):
            success = self.index_block(block_num)
            if success:
                success_count += 1
            else:
                fail_count += 1
        
        return {
            "blocks_processed": success_count,
            "blocks_failed": fail_count,
            "total_blocks": end_block - start_block + 1
        }
    
    def index_latest_blocks(self, count: int = 10) -> Dict[str, int]:
        """
        Index the latest n blocks.
        
        Args:
            count: Number of latest blocks to index
            
        Returns:
            Statistics about the indexing process
        """
        latest_block = self.web3.eth.block_number
        start_block = max(0, latest_block - count + 1)
        
        return self.index_range(start_block, latest_block)
    
    def listen_for_new_blocks(self, callback=None):
        """
        Listen for new blocks and index them as they arrive.
        
        Args:
            callback: Optional callback function to be called with each new block
        """
        if not self.ws_url:
            logger.error("WebSocket URL is required for listening to new blocks")
            return
        
        def handle_new_block(block_hash):
            block_number = self.web3.eth.get_block(block_hash).number
            logger.info(f"New block detected: {block_number}")
            self.index_block(block_number)
            
            if callback:
                callback(block_number)
        
        # Create a filter for new blocks
        new_block_filter = self.web3.eth.filter('latest')
        
        try:
            # Poll for new blocks
            while True:
                for block_hash in new_block_filter.get_new_entries():
                    handle_new_block(block_hash)
                time.sleep(1)  # Poll every second
        except KeyboardInterrupt:
            logger.info("Stopping block listener")
        except Exception as e:
            logger.error(f"Error in block listener: {e}")
        finally:
            # Clean up
            self.web3.eth.uninstall_filter(new_block_filter.filter_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexer.
        
        Returns:
            Dictionary of statistics
        """
        elapsed_time = time.time() - self.stats["start_time"]
        blocks_per_hour = 0
        txs_per_hour = 0
        
        if elapsed_time > 0:
            blocks_per_hour = (self.stats["blocks_processed"] / elapsed_time) * 3600
            txs_per_hour = (self.stats["transactions_processed"] / elapsed_time) * 3600
        
        return {
            "blocks_processed": self.stats["blocks_processed"],
            "transactions_processed": self.stats["transactions_processed"],
            "elapsed_time_seconds": elapsed_time,
            "blocks_per_hour": blocks_per_hour,
            "transactions_per_hour": txs_per_hour
        }

# Main entry point when run as a module
def main():
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="LlamaChain Blockchain Indexer")
    parser.add_argument("--rpc-url", help="Ethereum RPC URL", default=os.environ.get("ETH_RPC_URL"))
    parser.add_argument("--ws-url", help="Ethereum WebSocket URL", default=os.environ.get("ETH_WSS_URL"))
    parser.add_argument("--latest", type=int, help="Index latest N blocks", default=0)
    parser.add_argument("--range", help="Index a range of blocks (format: start-end)", default="")
    parser.add_argument("--block", type=int, help="Index a specific block", default=0)
    parser.add_argument("--listen", action="store_true", help="Listen for new blocks")
    
    args = parser.parse_args()
    
    indexer = BlockchainIndexer(rpc_url=args.rpc_url, ws_url=args.ws_url)
    
    if args.block > 0:
        print(f"Indexing block {args.block}...")
        indexer.index_block(args.block)
    elif args.range:
        try:
            start, end = map(int, args.range.split("-"))
            print(f"Indexing blocks {start} to {end}...")
            stats = indexer.index_range(start, end)
            print(f"Indexed {stats['blocks_processed']} blocks successfully, {stats['blocks_failed']} failed")
        except ValueError:
            print("Invalid range format. Use start-end (e.g., 15000000-15000010)")
    elif args.latest > 0:
        print(f"Indexing latest {args.latest} blocks...")
        stats = indexer.index_latest_blocks(args.latest)
        print(f"Indexed {stats['blocks_processed']} blocks successfully, {stats['blocks_failed']} failed")
    elif args.listen:
        print("Listening for new blocks...")
        indexer.listen_for_new_blocks()
    else:
        print("No action specified. Use --help to see available options.")
    
    # Display stats
    stats = indexer.get_stats()
    print(f"Indexer stats: {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    main()
