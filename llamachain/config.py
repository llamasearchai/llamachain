"""
Configuration management for the LlamaChain platform.
"""

import os
from enum import Enum
from typing import List, Optional, Union

from pydantic import AnyHttpUrl, BaseSettings, PostgresDsn, validator


class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class Settings(BaseSettings):
    """Application settings."""
    
    # Application settings
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    SECRET_KEY: str = "development-secret-change-me-in-production"
    
    # Database settings
    DATABASE_URL: PostgresDsn
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    JWT_SECRET: str = "development-jwt-secret-change-me-in-production"
    API_RATE_LIMIT: int = 100
    
    # Blockchain RPC URLs
    ETHEREUM_RPC_URL: str
    ETHEREUM_WS_URL: Optional[str] = None
    ETHEREUM_TESTNET_RPC_URL: Optional[str] = None
    
    SOLANA_RPC_URL: str
    SOLANA_WS_URL: Optional[str] = None
    SOLANA_TESTNET_RPC_URL: Optional[str] = None
    
    # Security settings
    SECURITY_SCAN_TIMEOUT: int = 60
    SECURITY_MAX_CONTRACT_SIZE: int = 500000
    
    # Analytics settings
    ANALYTICS_CACHE_TTL: int = 3600
    ANALYTICS_MAX_BLOCKS: int = 1000
    
    # Machine learning settings
    ML_MODEL_PATH: str = "./models"
    ML_EMBEDDINGS_DIMENSION: int = 768
    
    # Worker settings
    WORKER_CONCURRENCY: int = 4
    WORKER_QUEUE_URL: str = "redis://localhost:6379/0"
    
    # Web dashboard settings
    DASHBOARD_HOST: str = "0.0.0.0"
    DASHBOARD_PORT: int = 3000
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins from string to list."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = True


# Load settings from environment variables and .env file
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def get_blockchain_config(chain_id: str) -> dict:
    """
    Get configuration for a specific blockchain.
    
    Args:
        chain_id: The blockchain identifier (e.g., "ethereum", "solana")
        
    Returns:
        Dict with blockchain configuration
    """
    chain_id = chain_id.lower()
    
    if chain_id == "ethereum":
        return {
            "rpc_url": settings.ETHEREUM_RPC_URL,
            "ws_url": settings.ETHEREUM_WS_URL,
            "testnet_rpc_url": settings.ETHEREUM_TESTNET_RPC_URL,
            "chain_id": 1,  # Mainnet
            "testnet_chain_id": 5,  # Goerli
            "explorer_url": "https://etherscan.io",
            "testnet_explorer_url": "https://goerli.etherscan.io",
            "block_time": 12,  # seconds
        }
    elif chain_id == "solana":
        return {
            "rpc_url": settings.SOLANA_RPC_URL,
            "ws_url": settings.SOLANA_WS_URL,
            "testnet_rpc_url": settings.SOLANA_TESTNET_RPC_URL,
            "chain_id": "mainnet-beta",
            "testnet_chain_id": "devnet",
            "explorer_url": "https://explorer.solana.com",
            "testnet_explorer_url": "https://explorer.solana.com/?cluster=devnet",
            "block_time": 0.4,  # seconds
        }
    else:
        raise ValueError(f"Unsupported blockchain: {chain_id}")


def is_development() -> bool:
    """Check if the application is running in development mode."""
    return settings.ENVIRONMENT == Environment.DEVELOPMENT


def is_production() -> bool:
    """Check if the application is running in production mode."""
    return settings.ENVIRONMENT == Environment.PRODUCTION


def is_testing() -> bool:
    """Check if the application is running in test mode."""
    return settings.ENVIRONMENT == Environment.TESTING 