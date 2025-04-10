from pydantic_settings import BaseSettings
from pydantic import validator, AnyHttpUrl, PostgresDsn
from typing import List, Optional, Dict, Any, Union
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support and validation"""
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "CHANGE_THIS_TO_A_SECURE_RANDOM_KEY"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Database Configuration
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "llamachain"
    POSTGRES_PORT: str = "5432"
    DATABASE_URI: Optional[str] = None
    
    # SQLAlchemy async configuration
    SQLALCHEMY_DATABASE_URI: Optional[str] = None
    
    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        
        # If SQLite is preferred for development
        if os.environ.get("USE_SQLITE", "false").lower() == "true":
            sqlite_path = Path("./data/llamachain.db")
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite+aiosqlite:///{sqlite_path}"
            
        # Otherwise use PostgreSQL
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            port=values.get("POSTGRES_PORT"),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )
    
    # Blockchain Configuration
    ETH_RPC_URL: str = "https://mainnet.infura.io/v3/your-infura-key"
    ETH_WSS_URL: str = "wss://mainnet.infura.io/ws/v3/your-infura-key"
    ETH_CHAIN_ID: int = 1
    SOL_RPC_URL: str = "https://api.mainnet-beta.solana.com"
    
    # IPFS Configuration
    IPFS_HOST: str = "ipfs.infura.io"
    IPFS_PORT: int = 5001
    IPFS_PROTOCOL: str = "https"
    
    # ML Model Configuration
    ML_MODEL_PATH: str = "models/vulnerability_detector"
    USE_MLX: bool = False
    USE_JAX: bool = False
    
    @validator("USE_MLX", pre=True)
    def check_mlx_availability(cls, v: bool) -> bool:
        """Check if MLX is available (Apple Silicon)"""
        if v:
            try:
                import platform
                if platform.machine() == "arm64" and platform.system() == "Darwin":
                    try:
                        import mlx
                        return True
                    except ImportError:
                        return False
                return False
            except ImportError:
                return False
        return False
    
    @validator("USE_JAX", pre=True)
    def check_jax_availability(cls, v: bool) -> bool:
        """Check if JAX is available"""
        if v:
            try:
                import jax
                return True
            except ImportError:
                return False
        return False
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings() 