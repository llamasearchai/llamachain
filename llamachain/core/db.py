"""
Database utilities for the LlamaChain platform.

This module provides utilities for database operations.
"""

import os
import json
import logging
import sqlite3
from typing import Dict, List, Any, Optional, Union, Tuple

from llamachain.core.exceptions import DatabaseError
from llamachain.log import get_logger

# Get logger
logger = get_logger("llamachain.core.db")


class Database:
    """Simple database wrapper for storing blockchain data."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path or os.environ.get("DB_PATH", "data/llamachain.db")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.conn = None
        self.cursor = None
        
        # Connect to the database
        self._connect()
        
        # Create tables if they don't exist
        self._create_tables()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def _connect(self):
        """Connect to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise DatabaseError(f"Failed to connect to database: {str(e)}")
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            # Blocks table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS blocks (
                    chain_id TEXT,
                    block_number INTEGER,
                    block_hash TEXT,
                    parent_hash TEXT,
                    timestamp INTEGER,
                    transactions_count INTEGER,
                    gas_used INTEGER,
                    gas_limit INTEGER,
                    miner TEXT,
                    difficulty INTEGER,
                    total_difficulty INTEGER,
                    size INTEGER,
                    extra_data TEXT,
                    PRIMARY KEY (chain_id, block_number)
                )
            ''')
            
            # Transactions table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    chain_id TEXT,
                    tx_hash TEXT,
                    block_number INTEGER,
                    from_address TEXT,
                    to_address TEXT,
                    value TEXT,
                    gas INTEGER,
                    gas_price INTEGER,
                    gas_used INTEGER,
                    nonce INTEGER,
                    transaction_index INTEGER,
                    input_data TEXT,
                    timestamp INTEGER,
                    status INTEGER,
                    PRIMARY KEY (chain_id, tx_hash)
                )
            ''')
            
            # Contracts table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS contracts (
                    chain_id TEXT,
                    address TEXT,
                    creator_address TEXT,
                    creation_tx_hash TEXT,
                    bytecode TEXT,
                    abi TEXT,
                    name TEXT,
                    symbol TEXT,
                    contract_type TEXT,
                    verified BOOLEAN,
                    verified_at INTEGER,
                    source_code TEXT,
                    compiler_version TEXT,
                    optimization_used BOOLEAN,
                    optimization_runs INTEGER,
                    PRIMARY KEY (chain_id, address)
                )
            ''')
            
            # Security audits table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_audits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chain_id TEXT,
                    contract_address TEXT,
                    timestamp INTEGER,
                    audit_type TEXT,
                    vulnerabilities_count INTEGER,
                    audit_score INTEGER,
                    report TEXT,
                    UNIQUE (chain_id, contract_address, timestamp)
                )
            ''')
            
            # Vulnerabilities table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS vulnerabilities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audit_id INTEGER,
                    vulnerability_type TEXT,
                    severity TEXT,
                    title TEXT,
                    description TEXT,
                    line_number INTEGER,
                    function_name TEXT,
                    code_snippet TEXT,
                    recommendation TEXT,
                    FOREIGN KEY (audit_id) REFERENCES security_audits (id)
                )
            ''')
            
            # Alerts table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT,
                    chain_id TEXT,
                    address TEXT,
                    timestamp INTEGER,
                    title TEXT,
                    description TEXT,
                    severity TEXT,
                    data TEXT,
                    is_read BOOLEAN DEFAULT 0
                )
            ''')
            
            # Analytics cache table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE,
                    data TEXT,
                    created_at INTEGER,
                    expires_at INTEGER
                )
            ''')
            
            # Create indexes
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_blocks_timestamp ON blocks (timestamp)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_txs_from ON transactions (from_address)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_txs_to ON transactions (to_address)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_txs_timestamp ON transactions (timestamp)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts (timestamp)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_analytics_cache_expires ON analytics_cache (expires_at)')
            
            self.conn.commit()
            logger.debug("Database tables created/verified")
        except sqlite3.Error as e:
            logger.error(f"Error creating database tables: {e}")
            raise DatabaseError(f"Failed to create database tables: {str(e)}")
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.debug("Database connection closed")
    
    def store_block(self, block_data: Dict[str, Any]) -> bool:
        """
        Store block data in the database.
        
        Args:
            block_data: Block data as a dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract relevant fields from block_data
            chain_id = block_data.get("chain_id", "ethereum")
            block_number = block_data.get("number")
            block_hash = block_data.get("hash")
            parent_hash = block_data.get("parentHash")
            timestamp = block_data.get("timestamp")
            transactions = block_data.get("transactions", [])
            transactions_count = len(transactions)
            gas_used = block_data.get("gasUsed")
            gas_limit = block_data.get("gasLimit")
            miner = block_data.get("miner")
            difficulty = block_data.get("difficulty")
            total_difficulty = block_data.get("totalDifficulty")
            size = block_data.get("size")
            extra_data = block_data.get("extraData")
            
            # Insert into blocks table
            self.cursor.execute('''
                INSERT OR REPLACE INTO blocks (
                    chain_id, block_number, block_hash, parent_hash, timestamp,
                    transactions_count, gas_used, gas_limit, miner, difficulty,
                    total_difficulty, size, extra_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                chain_id, block_number, block_hash, parent_hash, timestamp,
                transactions_count, gas_used, gas_limit, miner, difficulty,
                total_difficulty, size, extra_data
            ))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error storing block data: {e}")
            return False
    
    def store_transaction(self, tx_data: Dict[str, Any]) -> bool:
        """
        Store transaction data in the database.
        
        Args:
            tx_data: Transaction data as a dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract relevant fields from tx_data
            chain_id = tx_data.get("chain_id", "ethereum")
            tx_hash = tx_data.get("hash")
            block_number = tx_data.get("blockNumber")
            from_address = tx_data.get("from")
            to_address = tx_data.get("to")
            value = str(tx_data.get("value", 0))  # Store as string to preserve precision
            gas = tx_data.get("gas")
            gas_price = tx_data.get("gasPrice")
            gas_used = tx_data.get("gasUsed")
            nonce = tx_data.get("nonce")
            transaction_index = tx_data.get("transactionIndex")
            input_data = tx_data.get("input")
            timestamp = tx_data.get("timestamp")
            status = tx_data.get("status")
            
            # Insert into transactions table
            self.cursor.execute('''
                INSERT OR REPLACE INTO transactions (
                    chain_id, tx_hash, block_number, from_address, to_address,
                    value, gas, gas_price, gas_used, nonce, transaction_index,
                    input_data, timestamp, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                chain_id, tx_hash, block_number, from_address, to_address,
                value, gas, gas_price, gas_used, nonce, transaction_index,
                input_data, timestamp, status
            ))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error storing transaction data: {e}")
            return False
    
    def store_contract(self, contract_data: Dict[str, Any]) -> bool:
        """
        Store contract data in the database.
        
        Args:
            contract_data: Contract data as a dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract relevant fields from contract_data
            chain_id = contract_data.get("chain_id", "ethereum")
            address = contract_data.get("address")
            creator_address = contract_data.get("creator_address")
            creation_tx_hash = contract_data.get("creation_tx_hash")
            bytecode = contract_data.get("bytecode")
            abi = json.dumps(contract_data.get("abi", {}))
            name = contract_data.get("name")
            symbol = contract_data.get("symbol")
            contract_type = contract_data.get("contract_type")
            verified = contract_data.get("verified", False)
            verified_at = contract_data.get("verified_at")
            source_code = contract_data.get("source_code")
            compiler_version = contract_data.get("compiler_version")
            optimization_used = contract_data.get("optimization_used", False)
            optimization_runs = contract_data.get("optimization_runs")
            
            # Insert into contracts table
            self.cursor.execute('''
                INSERT OR REPLACE INTO contracts (
                    chain_id, address, creator_address, creation_tx_hash,
                    bytecode, abi, name, symbol, contract_type, verified,
                    verified_at, source_code, compiler_version,
                    optimization_used, optimization_runs
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                chain_id, address, creator_address, creation_tx_hash,
                bytecode, abi, name, symbol, contract_type, verified,
                verified_at, source_code, compiler_version,
                optimization_used, optimization_runs
            ))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error storing contract data: {e}")
            return False
    
    def store_audit(self, audit_data: Dict[str, Any]) -> int:
        """
        Store security audit data in the database.
        
        Args:
            audit_data: Audit data as a dictionary
            
        Returns:
            Audit ID if successful, -1 otherwise
        """
        try:
            # Extract relevant fields from audit_data
            chain_id = audit_data.get("chain_id", "ethereum")
            contract_address = audit_data.get("contract_address")
            timestamp = audit_data.get("timestamp")
            audit_type = audit_data.get("audit_type")
            vulnerabilities_count = audit_data.get("vulnerabilities_count", 0)
            audit_score = audit_data.get("audit_score")
            report = json.dumps(audit_data.get("report", {}))
            
            # Insert into security_audits table
            self.cursor.execute('''
                INSERT OR REPLACE INTO security_audits (
                    chain_id, contract_address, timestamp, audit_type,
                    vulnerabilities_count, audit_score, report
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                chain_id, contract_address, timestamp, audit_type,
                vulnerabilities_count, audit_score, report
            ))
            
            # Get the ID of the inserted audit
            audit_id = self.cursor.lastrowid
            
            # Store vulnerabilities if provided
            vulnerabilities = audit_data.get("vulnerabilities", [])
            for vuln in vulnerabilities:
                self.store_vulnerability(audit_id, vuln)
            
            self.conn.commit()
            return audit_id
        except sqlite3.Error as e:
            logger.error(f"Error storing audit data: {e}")
            return -1
    
    def store_vulnerability(self, audit_id: int, vulnerability_data: Dict[str, Any]) -> bool:
        """
        Store vulnerability data in the database.
        
        Args:
            audit_id: ID of the associated audit
            vulnerability_data: Vulnerability data as a dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract relevant fields from vulnerability_data
            vulnerability_type = vulnerability_data.get("type")
            severity = vulnerability_data.get("severity")
            title = vulnerability_data.get("title")
            description = vulnerability_data.get("description")
            line_number = vulnerability_data.get("line_number")
            function_name = vulnerability_data.get("function_name")
            code_snippet = vulnerability_data.get("code_snippet")
            recommendation = vulnerability_data.get("recommendation")
            
            # Insert into vulnerabilities table
            self.cursor.execute('''
                INSERT INTO vulnerabilities (
                    audit_id, vulnerability_type, severity, title,
                    description, line_number, function_name,
                    code_snippet, recommendation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                audit_id, vulnerability_type, severity, title,
                description, line_number, function_name,
                code_snippet, recommendation
            ))
            
            return True
        except sqlite3.Error as e:
            logger.error(f"Error storing vulnerability data: {e}")
            return False
    
    def store_alert(self, alert_data: Dict[str, Any]) -> int:
        """
        Store alert data in the database.
        
        Args:
            alert_data: Alert data as a dictionary
            
        Returns:
            Alert ID if successful, -1 otherwise
        """
        try:
            # Extract relevant fields from alert_data
            alert_type = alert_data.get("alert_type")
            chain_id = alert_data.get("chain_id")
            address = alert_data.get("address")
            timestamp = alert_data.get("timestamp")
            title = alert_data.get("title")
            description = alert_data.get("description")
            severity = alert_data.get("severity")
            data = json.dumps(alert_data.get("data", {}))
            
            # Insert into alerts table
            self.cursor.execute('''
                INSERT INTO alerts (
                    alert_type, chain_id, address, timestamp,
                    title, description, severity, data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert_type, chain_id, address, timestamp,
                title, description, severity, data
            ))
            
            # Get the ID of the inserted alert
            alert_id = self.cursor.lastrowid
            
            self.conn.commit()
            return alert_id
        except sqlite3.Error as e:
            logger.error(f"Error storing alert data: {e}")
            return -1
    
    def get_block(self, chain_id: str, block_number: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a block by its number.
        
        Args:
            chain_id: Blockchain ID
            block_number: Block number
            
        Returns:
            Block data as a dictionary, or None if not found
        """
        try:
            self.cursor.execute('''
                SELECT * FROM blocks WHERE chain_id = ? AND block_number = ?
            ''', (chain_id, block_number))
            
            row = self.cursor.fetchone()
            if row:
                return dict(row)
            return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving block data: {e}")
            return None
    
    def get_transaction(self, chain_id: str, tx_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a transaction by its hash.
        
        Args:
            chain_id: Blockchain ID
            tx_hash: Transaction hash
            
        Returns:
            Transaction data as a dictionary, or None if not found
        """
        try:
            self.cursor.execute('''
                SELECT * FROM transactions WHERE chain_id = ? AND tx_hash = ?
            ''', (chain_id, tx_hash))
            
            row = self.cursor.fetchone()
            if row:
                return dict(row)
            return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving transaction data: {e}")
            return None
    
    def get_transactions_by_address(
        self, 
        chain_id: str, 
        address: str, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve transactions by address.
        
        Args:
            chain_id: Blockchain ID
            address: Address
            limit: Maximum number of transactions to retrieve
            offset: Number of transactions to skip
            
        Returns:
            List of transaction data dictionaries
        """
        try:
            self.cursor.execute('''
                SELECT * FROM transactions 
                WHERE chain_id = ? AND (from_address = ? OR to_address = ?)
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            ''', (chain_id, address, address, limit, offset))
            
            rows = self.cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving transactions by address: {e}")
            return []
    
    def get_contract(self, chain_id: str, address: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a contract by its address.
        
        Args:
            chain_id: Blockchain ID
            address: Contract address
            
        Returns:
            Contract data as a dictionary, or None if not found
        """
        try:
            self.cursor.execute('''
                SELECT * FROM contracts WHERE chain_id = ? AND address = ?
            ''', (chain_id, address))
            
            row = self.cursor.fetchone()
            if row:
                contract_data = dict(row)
                # Parse JSON string back to dictionary
                if contract_data.get("abi"):
                    try:
                        contract_data["abi"] = json.loads(contract_data["abi"])
                    except json.JSONDecodeError:
                        contract_data["abi"] = {}
                return contract_data
            return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving contract data: {e}")
            return None
    
    def get_latest_audit(self, chain_id: str, contract_address: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the latest security audit for a contract.
        
        Args:
            chain_id: Blockchain ID
            contract_address: Contract address
            
        Returns:
            Audit data as a dictionary, or None if not found
        """
        try:
            self.cursor.execute('''
                SELECT * FROM security_audits 
                WHERE chain_id = ? AND contract_address = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (chain_id, contract_address))
            
            row = self.cursor.fetchone()
            if row:
                audit_data = dict(row)
                # Parse JSON string back to dictionary
                if audit_data.get("report"):
                    try:
                        audit_data["report"] = json.loads(audit_data["report"])
                    except json.JSONDecodeError:
                        audit_data["report"] = {}
                
                # Get vulnerabilities for this audit
                audit_data["vulnerabilities"] = self.get_vulnerabilities(audit_data["id"])
                
                return audit_data
            return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving audit data: {e}")
            return None
    
    def get_vulnerabilities(self, audit_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve vulnerabilities for an audit.
        
        Args:
            audit_id: ID of the audit
            
        Returns:
            List of vulnerability data dictionaries
        """
        try:
            self.cursor.execute('''
                SELECT * FROM vulnerabilities WHERE audit_id = ?
            ''', (audit_id,))
            
            rows = self.cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving vulnerabilities: {e}")
            return []
    
    def get_recent_alerts(
        self, 
        limit: int = 10, 
        alert_type: Optional[str] = None,
        severity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent alerts.
        
        Args:
            limit: Maximum number of alerts to retrieve
            alert_type: Filter by alert type
            severity: Filter by severity
            
        Returns:
            List of alert data dictionaries
        """
        try:
            query = "SELECT * FROM alerts"
            params = []
            
            where_clauses = []
            if alert_type:
                where_clauses.append("alert_type = ?")
                params.append(alert_type)
            
            if severity:
                where_clauses.append("severity = ?")
                params.append(severity)
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            self.cursor.execute(query, tuple(params))
            
            rows = self.cursor.fetchall()
            alerts = []
            for row in rows:
                alert_data = dict(row)
                # Parse JSON string back to dictionary
                if alert_data.get("data"):
                    try:
                        alert_data["data"] = json.loads(alert_data["data"])
                    except json.JSONDecodeError:
                        alert_data["data"] = {}
                alerts.append(alert_data)
            
            return alerts
        except sqlite3.Error as e:
            logger.error(f"Error retrieving alerts: {e}")
            return []
    
    def mark_alert_as_read(self, alert_id: int) -> bool:
        """
        Mark an alert as read.
        
        Args:
            alert_id: ID of the alert
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cursor.execute('''
                UPDATE alerts SET is_read = 1 WHERE id = ?
            ''', (alert_id,))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error marking alert as read: {e}")
            return False
    
    def get_analytics_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached analytics data.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached data as a dictionary, or None if not found or expired
        """
        try:
            current_time = int(time.time())
            
            self.cursor.execute('''
                SELECT * FROM analytics_cache 
                WHERE cache_key = ? AND expires_at > ?
            ''', (cache_key, current_time))
            
            row = self.cursor.fetchone()
            if row:
                cache_data = dict(row)
                # Parse JSON string back to dictionary
                if cache_data.get("data"):
                    try:
                        return json.loads(cache_data["data"])
                    except json.JSONDecodeError:
                        return None
            
            return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving analytics cache: {e}")
            return None
    
    def set_analytics_cache(self, cache_key: str, data: Dict[str, Any], ttl_seconds: int = 3600) -> bool:
        """
        Cache analytics data.
        
        Args:
            cache_key: Cache key
            data: Data to cache
            ttl_seconds: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            current_time = int(time.time())
            expires_at = current_time + ttl_seconds
            
            # Convert data to JSON string
            data_json = json.dumps(data)
            
            self.cursor.execute('''
                INSERT OR REPLACE INTO analytics_cache (
                    cache_key, data, created_at, expires_at
                ) VALUES (?, ?, ?, ?)
            ''', (cache_key, data_json, current_time, expires_at))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error setting analytics cache: {e}")
            return False
    
    def clear_expired_cache(self) -> int:
        """
        Clear expired cached data.
        
        Returns:
            Number of cleared cache entries
        """
        try:
            current_time = int(time.time())
            
            self.cursor.execute('''
                DELETE FROM analytics_cache WHERE expires_at < ?
            ''', (current_time,))
            
            num_cleared = self.cursor.rowcount
            self.conn.commit()
            return num_cleared
        except sqlite3.Error as e:
            logger.error(f"Error clearing expired cache: {e}")
            return 0 