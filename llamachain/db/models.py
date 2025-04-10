"""
Database models for the LlamaChain platform.
"""

import enum
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, 
    ForeignKey, Text, Enum, JSON, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from llamachain.db.session import Base


class BlockchainEnum(enum.Enum):
    """Enum for supported blockchains."""
    ethereum = "ethereum"
    solana = "solana"


class VulnerabilitySeverity(enum.Enum):
    """Enum for vulnerability severity levels."""
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    info = "info"


class User(Base):
    """User model for authentication and preferences."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", cascade="all, delete-orphan")
    watchlist = relationship("WatchlistItem", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User {self.username}>"


class ApiKey(Base):
    """API key model for API authentication."""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key = Column(String(64), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    def __repr__(self):
        return f"<ApiKey {self.name}>"


class UserPreference(Base):
    """User preference model for storing user settings."""
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key = Column(String(50), nullable=False)
    value = Column(Text, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="preferences")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("user_id", "key", name="uq_user_preference"),
    )
    
    def __repr__(self):
        return f"<UserPreference {self.key}={self.value}>"


class WatchlistItem(Base):
    """Watchlist item model for tracking addresses and contracts."""
    __tablename__ = "watchlist_items"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    blockchain = Column(Enum(BlockchainEnum), nullable=False)
    address = Column(String(255), nullable=False)
    name = Column(String(100), nullable=True)
    is_contract = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="watchlist")
    alerts = relationship("Alert", back_populates="watchlist_item", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("user_id", "blockchain", "address", name="uq_watchlist_item"),
    )
    
    def __repr__(self):
        return f"<WatchlistItem {self.blockchain.value}:{self.address}>"


class Contract(Base):
    """Smart contract model for storing contract information."""
    __tablename__ = "contracts"
    
    id = Column(Integer, primary_key=True)
    blockchain = Column(Enum(BlockchainEnum), nullable=False)
    address = Column(String(255), nullable=False)
    name = Column(String(100), nullable=True)
    source_code = Column(Text, nullable=True)
    bytecode = Column(Text, nullable=True)
    abi = Column(JSON, nullable=True)
    compiler_version = Column(String(50), nullable=True)
    optimization_enabled = Column(Boolean, nullable=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    audits = relationship("Audit", back_populates="contract", cascade="all, delete-orphan")
    vulnerabilities = relationship("Vulnerability", back_populates="contract", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("blockchain", "address", name="uq_contract"),
    )
    
    def __repr__(self):
        return f"<Contract {self.blockchain.value}:{self.address}>"


class Audit(Base):
    """Audit model for storing contract audit information."""
    __tablename__ = "audits"
    
    id = Column(Integer, primary_key=True)
    contract_id = Column(Integer, ForeignKey("contracts.id"), nullable=False)
    timestamp = Column(DateTime, default=func.now())
    auditor = Column(String(100), nullable=False)
    score = Column(Float, nullable=True)
    report_data = Column(JSON, nullable=True)
    
    # Relationships
    contract = relationship("Contract", back_populates="audits")
    
    def __repr__(self):
        return f"<Audit {self.id} for contract {self.contract_id}>"


class Vulnerability(Base):
    """Vulnerability model for storing contract vulnerabilities."""
    __tablename__ = "vulnerabilities"
    
    id = Column(Integer, primary_key=True)
    contract_id = Column(Integer, ForeignKey("contracts.id"), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    severity = Column(Enum(VulnerabilitySeverity), nullable=False)
    type = Column(String(100), nullable=False)
    line_number = Column(Integer, nullable=True)
    function_name = Column(String(100), nullable=True)
    detector = Column(String(50), nullable=False)  # static, ml, manual
    confidence = Column(Float, nullable=False, default=1.0)
    details = Column(JSON, nullable=True)
    is_fixed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    contract = relationship("Contract", back_populates="vulnerabilities")
    
    def __repr__(self):
        return f"<Vulnerability {self.title} ({self.severity.value})>"


class Alert(Base):
    """Alert model for storing security and anomaly alerts."""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True)
    blockchain = Column(Enum(BlockchainEnum), nullable=False)
    timestamp = Column(DateTime, default=func.now())
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    severity = Column(String(20), nullable=False)  # critical, high, medium, low, info
    category = Column(String(50), nullable=False)  # security, anomaly, price, etc.
    source = Column(String(50), nullable=False)  # system, user, external
    address = Column(String(255), nullable=True)
    transaction_hash = Column(String(255), nullable=True)
    block_number = Column(Integer, nullable=True)
    details = Column(JSON, nullable=True)
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    watchlist_item_id = Column(Integer, ForeignKey("watchlist_items.id"), nullable=True)
    
    # Relationships
    watchlist_item = relationship("WatchlistItem", back_populates="alerts")
    
    # Indexes
    __table_args__ = (
        Index("ix_alerts_blockchain_timestamp", "blockchain", "timestamp"),
        Index("ix_alerts_severity", "severity"),
        Index("ix_alerts_category", "category"),
        Index("ix_alerts_address", "address"),
    )
    
    def __repr__(self):
        return f"<Alert {self.title} ({self.severity})>"


class Transaction(Base):
    """Transaction model for storing cached transaction data."""
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True)
    blockchain = Column(Enum(BlockchainEnum), nullable=False)
    hash = Column(String(255), nullable=False)
    block_number = Column(Integer, nullable=True)
    timestamp = Column(DateTime, nullable=True)
    from_address = Column(String(255), nullable=False)
    to_address = Column(String(255), nullable=True)
    value = Column(String(100), nullable=False)  # String to handle large numbers
    gas_price = Column(String(100), nullable=True)
    gas_used = Column(Integer, nullable=True)
    status = Column(Boolean, nullable=True)
    data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.now())
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("blockchain", "hash", name="uq_transaction"),
        Index("ix_transactions_from_address", "blockchain", "from_address"),
        Index("ix_transactions_to_address", "blockchain", "to_address"),
        Index("ix_transactions_block_number", "blockchain", "block_number"),
    )
    
    def __repr__(self):
        return f"<Transaction {self.blockchain.value}:{self.hash}>"


class Block(Base):
    """Block model for storing cached block data."""
    __tablename__ = "blocks"
    
    id = Column(Integer, primary_key=True)
    blockchain = Column(Enum(BlockchainEnum), nullable=False)
    number = Column(Integer, nullable=False)
    hash = Column(String(255), nullable=False)
    timestamp = Column(DateTime, nullable=True)
    parent_hash = Column(String(255), nullable=True)
    transaction_count = Column(Integer, nullable=True)
    data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.now())
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("blockchain", "number", name="uq_block_number"),
        UniqueConstraint("blockchain", "hash", name="uq_block_hash"),
    )
    
    def __repr__(self):
        return f"<Block {self.blockchain.value}:{self.number}>"


class ApiUsage(Base):
    """API usage model for tracking API usage."""
    __tablename__ = "api_usage"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=True)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time = Column(Float, nullable=False)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(255), nullable=True)
    timestamp = Column(DateTime, default=func.now())
    
    # Indexes
    __table_args__ = (
        Index("ix_api_usage_user_id", "user_id"),
        Index("ix_api_usage_api_key_id", "api_key_id"),
        Index("ix_api_usage_endpoint", "endpoint"),
        Index("ix_api_usage_timestamp", "timestamp"),
    )
    
    def __repr__(self):
        return f"<ApiUsage {self.endpoint} at {self.timestamp}>" 