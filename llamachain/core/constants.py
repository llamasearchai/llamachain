"""
Core constants for the LlamaChain platform.

This module defines constants used throughout the application.
"""

from enum import Enum, auto
from typing import Dict, List, Set


class BlockchainType(str, Enum):
    """Supported blockchain types."""
    ETHEREUM = "ethereum"
    SOLANA = "solana"


class AuditSeverityLevel(str, Enum):
    """Audit severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class AlertType(str, Enum):
    """Types of alerts."""
    SECURITY = "security"
    PRICE = "price"
    VOLUME = "volume"
    TRANSACTION = "transaction"
    WHALE = "whale"
    CUSTOM = "custom"


class BlockchainNetwork(str, Enum):
    """Blockchain networks."""
    # Ethereum networks
    ETH_MAINNET = "eth_mainnet"
    ETH_GOERLI = "eth_goerli"
    ETH_SEPOLIA = "eth_sepolia"
    
    # Solana networks
    SOL_MAINNET = "sol_mainnet"
    SOL_DEVNET = "sol_devnet"
    SOL_TESTNET = "sol_testnet"


class ContractType(str, Enum):
    """Smart contract types."""
    ERC20 = "erc20"
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    CUSTOM = "custom"
    DEFI = "defi"
    DEX = "dex"
    LENDING = "lending"
    GOVERNANCE = "governance"
    BRIDGE = "bridge"
    ORACLE = "oracle"


class VulnerabilityType(str, Enum):
    """Vulnerability types."""
    REENTRANCY = "reentrancy"
    INTEGER_OVERFLOW = "integer_overflow"
    INTEGER_UNDERFLOW = "integer_underflow"
    ACCESS_CONTROL = "access_control"
    FRONT_RUNNING = "front_running"
    TIMESTAMP_DEPENDENCE = "timestamp_dependence"
    DOS = "denial_of_service"
    LOGIC_ERROR = "logic_error"
    ORACLE_MANIPULATION = "oracle_manipulation"
    FLASH_LOAN_ATTACK = "flash_loan_attack"
    WRONG_CONSTRUCTOR_NAME = "wrong_constructor_name"
    UNINITIALIZED_STORAGE = "uninitialized_storage"
    COMPILER_VERSION = "compiler_version"
    UNCHECKED_RETURN = "unchecked_return"
    LOCKED_ETHER = "locked_ether"
    ARBITRARY_SEND = "arbitrary_send"
    RACE_CONDITION = "race_condition"
    OUTDATED_COMPILER = "outdated_compiler"
    SHADOWING = "shadowing"
    OTHER = "other"


# Shorthand access to enum values
BLOCKCHAIN_TYPES = [blockchain.value for blockchain in BlockchainType]
AUDIT_SEVERITY_LEVELS = [level.value for level in AuditSeverityLevel]
ALERT_TYPES = [alert_type.value for alert_type in AlertType]
BLOCKCHAIN_NETWORKS = [network.value for network in BlockchainNetwork]
CONTRACT_TYPES = [contract_type.value for contract_type in ContractType]
VULNERABILITY_TYPES = [vuln_type.value for vuln_type in VulnerabilityType]

# Mapping from blockchain type to networks
BLOCKCHAIN_NETWORKS_MAP: Dict[str, List[str]] = {
    BlockchainType.ETHEREUM.value: [
        BlockchainNetwork.ETH_MAINNET.value,
        BlockchainNetwork.ETH_GOERLI.value,
        BlockchainNetwork.ETH_SEPOLIA.value,
    ],
    BlockchainType.SOLANA.value: [
        BlockchainNetwork.SOL_MAINNET.value,
        BlockchainNetwork.SOL_DEVNET.value,
        BlockchainNetwork.SOL_TESTNET.value,
    ],
}

# Default API pagination settings
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# Gas price thresholds (in gwei)
GAS_PRICE_LOW = 30
GAS_PRICE_MEDIUM = 60
GAS_PRICE_HIGH = 100

# File extensions for smart contracts
SOLIDITY_EXTENSIONS = {'.sol'}
RUST_EXTENSIONS = {'.rs'}

# Timestamp constants
SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 3600
SECONDS_PER_MINUTE = 60

# Transaction status
class TransactionStatus(str, Enum):
    """Transaction status values."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    DROPPED = "dropped"
    REPLACED = "replaced"
    UNKNOWN = "unknown"

# Feature flags
FEATURE_FLAGS = {
    "ENABLE_ML_FEATURES": False,
    "ENABLE_SECURITY_AUDIT": True,
    "ENABLE_PRICE_TRACKING": True,
    "ENABLE_ALERTS": True,
    "ENABLE_ADVANCED_ANALYTICS": False,
}

# Rate limiting constants
API_RATE_LIMIT_DEFAULT = 100  # requests per minute
API_RATE_LIMIT_AUTHENTICATED = 500  # requests per minute 