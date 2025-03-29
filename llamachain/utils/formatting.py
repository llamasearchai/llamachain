"""
Formatting utilities for the LlamaChain platform.

This module provides functions for formatting blockchain data like addresses, values, and timestamps.
"""

import datetime
from decimal import Decimal
from typing import Union, Optional


def format_wei_to_eth(wei_value: Union[int, str]) -> str:
    """
    Convert wei to ETH with proper formatting.
    
    Args:
        wei_value: Amount in wei (as int or string)
        
    Returns:
        Formatted ETH value as string
    """
    # Convert to Decimal for precision
    if isinstance(wei_value, str):
        wei_decimal = Decimal(wei_value)
    else:
        wei_decimal = Decimal(str(wei_value))
    
    # 1 ETH = 10^18 wei
    eth_value = wei_decimal / Decimal('1000000000000000000')
    
    # Format with up to 18 decimal places, but remove trailing zeros
    formatted = f"{eth_value:.18f}".rstrip('0').rstrip('.')
    
    return f"{formatted} ETH"


def format_lamports_to_sol(lamports: Union[int, str]) -> str:
    """
    Convert lamports to SOL with proper formatting.
    
    Args:
        lamports: Amount in lamports (as int or string)
        
    Returns:
        Formatted SOL value as string
    """
    # Convert to Decimal for precision
    if isinstance(lamports, str):
        lamports_decimal = Decimal(lamports)
    else:
        lamports_decimal = Decimal(str(lamports))
    
    # 1 SOL = 10^9 lamports
    sol_value = lamports_decimal / Decimal('1000000000')
    
    # Format with up to 9 decimal places, but remove trailing zeros
    formatted = f"{sol_value:.9f}".rstrip('0').rstrip('.')
    
    return f"{formatted} SOL"


def format_timestamp(timestamp: Union[int, float, str, datetime.datetime], 
                    format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a timestamp to human-readable string.
    
    Args:
        timestamp: Unix timestamp (seconds since epoch), 
                  or ISO format string,
                  or datetime object
        format_str: Format string for datetime.strftime
        
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, datetime.datetime):
        dt = timestamp
    elif isinstance(timestamp, (int, float)):
        dt = datetime.datetime.fromtimestamp(timestamp)
    elif isinstance(timestamp, str):
        # Try to parse as ISO format
        try:
            dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            # If not ISO format, try to parse as timestamp
            try:
                dt = datetime.datetime.fromtimestamp(float(timestamp))
            except ValueError:
                raise ValueError(f"Could not parse timestamp: {timestamp}")
    else:
        raise TypeError(f"Unsupported timestamp type: {type(timestamp)}")
    
    return dt.strftime(format_str)


def shorten_address(address: str, chars: int = 4) -> str:
    """
    Shorten an address for display by keeping the first and last few characters.
    
    Args:
        address: The address to shorten
        chars: Number of characters to keep at start and end
        
    Returns:
        Shortened address string
    """
    if not address:
        return ""
    
    if len(address) <= chars * 2 + 2:
        return address
    
    # For Ethereum addresses, keep the '0x' prefix
    if address.startswith("0x"):
        return f"{address[:chars+2]}...{address[-chars:]}"
    
    return f"{address[:chars]}...{address[-chars:]}"


def format_gas(gas: Union[int, str]) -> str:
    """
    Format gas value with commas for readability.
    
    Args:
        gas: Gas value
        
    Returns:
        Formatted gas string
    """
    if isinstance(gas, str):
        try:
            gas = int(gas)
        except ValueError:
            return gas
    
    return f"{gas:,}"


def format_currency(amount: Union[int, float, str, Decimal], 
                   symbol: str = "$", 
                   decimals: int = 2) -> str:
    """
    Format a currency amount.
    
    Args:
        amount: Amount to format
        symbol: Currency symbol
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    # Convert to Decimal for precision
    if isinstance(amount, str):
        try:
            amount = Decimal(amount)
        except:
            return f"{symbol}{amount}"
    elif isinstance(amount, (int, float)):
        amount = Decimal(str(amount))
    
    # Format with commas as thousands separator
    formatted = f"{float(amount):,.{decimals}f}"
    
    return f"{symbol}{formatted}"


def format_percentage(value: Union[float, str, Decimal], 
                     decimals: int = 2) -> str:
    """
    Format a value as a percentage.
    
    Args:
        value: Value to format (0.1 for 10%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return f"{value}%"
    
    return f"{value * 100:.{decimals}f}%"


def format_file_size(size_bytes: Union[int, float]) -> str:
    """
    Format a file size in bytes to a human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted file size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    
    kb = size_bytes / 1024
    if kb < 1024:
        return f"{kb:.2f} KB"
    
    mb = kb / 1024
    if mb < 1024:
        return f"{mb:.2f} MB"
    
    gb = mb / 1024
    return f"{gb:.2f} GB" 