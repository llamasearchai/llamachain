"""
Custom exceptions for the LlamaChain platform.

This module defines custom exception classes used throughout the application.
"""

from typing import Optional, Any, Dict


class LlamaChainError(Exception):
    """Base exception class for all LlamaChain exceptions."""
    
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            code: Error code for categorization
            details: Additional error details
        """
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the exception to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the exception
        """
        result = {
            "error": True,
            "message": self.message
        }
        
        if self.code:
            result["code"] = self.code
            
        if self.details:
            result["details"] = self.details
            
        return result


class BlockchainError(LlamaChainError):
    """Exception raised for blockchain-related errors."""
    
    def __init__(
        self, 
        message: str, 
        blockchain: Optional[str] = None, 
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the blockchain exception.
        
        Args:
            message: Error message
            blockchain: Blockchain identifier (e.g., "ethereum", "solana")
            code: Error code for categorization
            details: Additional error details
        """
        self.blockchain = blockchain
        details = details or {}
        
        if blockchain:
            details["blockchain"] = blockchain
            
        super().__init__(message, code=code, details=details)


class SecurityError(LlamaChainError):
    """Exception raised for security-related errors."""
    
    def __init__(
        self, 
        message: str, 
        severity: Optional[str] = None,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the security exception.
        
        Args:
            message: Error message
            severity: Severity level (e.g., "critical", "high", "medium", "low")
            code: Error code for categorization
            details: Additional error details
        """
        self.severity = severity
        details = details or {}
        
        if severity:
            details["severity"] = severity
            
        super().__init__(message, code=code, details=details)


class APIError(LlamaChainError):
    """Exception raised for API-related errors."""
    
    def __init__(
        self, 
        message: str, 
        status_code: int = 500,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the API exception.
        
        Args:
            message: Error message
            status_code: HTTP status code
            code: Error code for categorization
            details: Additional error details
        """
        self.status_code = status_code
        details = details or {}
        details["status_code"] = status_code
            
        super().__init__(message, code=code, details=details)


class ConfigError(LlamaChainError):
    """Exception raised for configuration-related errors."""
    
    def __init__(
        self, 
        message: str, 
        param: Optional[str] = None,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the configuration exception.
        
        Args:
            message: Error message
            param: Name of the configuration parameter that caused the error
            code: Error code for categorization
            details: Additional error details
        """
        self.param = param
        details = details or {}
        
        if param:
            details["param"] = param
            
        super().__init__(message, code=code, details=details)


class DatabaseError(LlamaChainError):
    """Exception raised for database-related errors."""
    
    def __init__(
        self, 
        message: str, 
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the database exception.
        
        Args:
            message: Error message
            code: Error code for categorization
            details: Additional error details
        """
        super().__init__(message, code=code or "database_error", details=details)


class AuthenticationError(LlamaChainError):
    """Exception raised for authentication-related errors."""
    
    def __init__(
        self, 
        message: str = "Authentication failed", 
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the authentication exception.
        
        Args:
            message: Error message
            code: Error code for categorization
            details: Additional error details
        """
        super().__init__(message, code=code or "authentication_error", details=details)


class AuthorizationError(LlamaChainError):
    """Exception raised for authorization-related errors."""
    
    def __init__(
        self, 
        message: str = "Not authorized", 
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the authorization exception.
        
        Args:
            message: Error message
            code: Error code for categorization
            details: Additional error details
        """
        super().__init__(message, code=code or "authorization_error", details=details)


class ValidationError(LlamaChainError):
    """Exception raised for validation-related errors."""
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the validation exception.
        
        Args:
            message: Error message
            field: Name of the field that failed validation
            code: Error code for categorization
            details: Additional error details
        """
        self.field = field
        details = details or {}
        
        if field:
            details["field"] = field
            
        super().__init__(message, code=code or "validation_error", details=details)


class ServiceUnavailableError(LlamaChainError):
    """Exception raised when a service is unavailable."""
    
    def __init__(
        self, 
        message: str = "Service unavailable", 
        service: Optional[str] = None,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the service unavailable exception.
        
        Args:
            message: Error message
            service: Name of the unavailable service
            code: Error code for categorization
            details: Additional error details
        """
        self.service = service
        details = details or {}
        
        if service:
            details["service"] = service
            
        super().__init__(message, code=code or "service_unavailable", details=details) 