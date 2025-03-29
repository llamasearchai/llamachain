"""
Time utilities for the LlamaChain platform.

This module provides functions for handling timestamps and time-related operations.
"""

import time
import datetime
from typing import Optional, Union, Tuple, List


def get_utc_now() -> datetime.datetime:
    """
    Get the current UTC time as a datetime object.
    
    Returns:
        Current UTC datetime
    """
    return datetime.datetime.now(datetime.timezone.utc)


def get_timestamp() -> int:
    """
    Get the current UTC timestamp in seconds.
    
    Returns:
        Current UTC timestamp as an integer
    """
    return int(time.time())


def get_timestamp_ms() -> int:
    """
    Get the current UTC timestamp in milliseconds.
    
    Returns:
        Current UTC timestamp in milliseconds as an integer
    """
    return int(time.time() * 1000)


def parse_timestamp(timestamp: Union[int, float, str, datetime.datetime]) -> datetime.datetime:
    """
    Parse a timestamp into a datetime object.
    
    Args:
        timestamp: Unix timestamp (seconds since epoch), 
                  or ISO format string,
                  or datetime object
                  
    Returns:
        Datetime object
    """
    if isinstance(timestamp, datetime.datetime):
        # Already a datetime object
        return timestamp
    elif isinstance(timestamp, (int, float)):
        # Unix timestamp
        return datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    elif isinstance(timestamp, str):
        # Try to parse as ISO format
        try:
            return datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            # If not ISO format, try to parse as timestamp
            try:
                return datetime.datetime.fromtimestamp(float(timestamp), tz=datetime.timezone.utc)
            except ValueError:
                raise ValueError(f"Could not parse timestamp: {timestamp}")
    else:
        raise TypeError(f"Unsupported timestamp type: {type(timestamp)}")


def datetime_to_timestamp(dt: datetime.datetime) -> int:
    """
    Convert a datetime object to a Unix timestamp.
    
    Args:
        dt: Datetime object
        
    Returns:
        Unix timestamp as an integer
    """
    return int(dt.timestamp())


def datetime_to_iso(dt: datetime.datetime) -> str:
    """
    Convert a datetime object to ISO 8601 format.
    
    Args:
        dt: Datetime object
        
    Returns:
        ISO 8601 formatted string
    """
    return dt.isoformat()


def get_time_delta(time_str: str) -> datetime.timedelta:
    """
    Parse a time string into a timedelta object.
    
    Args:
        time_str: Time string in format like "1d", "2h", "30m", "45s"
                or a combination like "1d12h30m"
                
    Returns:
        Timedelta object
    """
    if not time_str:
        raise ValueError("Time string cannot be empty")
    
    # Replace common variations
    time_str = time_str.lower().replace(" ", "")
    
    # Initialize timedelta components
    days = 0
    hours = 0
    minutes = 0
    seconds = 0
    
    # Use remaining string for processing
    remaining = time_str
    
    # Extract days
    if 'd' in remaining:
        parts = remaining.split('d')
        days = int(parts[0])
        remaining = parts[1]
    
    # Extract hours
    if 'h' in remaining:
        parts = remaining.split('h')
        hours = int(parts[0])
        remaining = parts[1]
    
    # Extract minutes
    if 'm' in remaining:
        # Make sure this is not milliseconds
        if not remaining.endswith('ms'):
            parts = remaining.split('m')
            minutes = int(parts[0])
            remaining = parts[1]
    
    # Extract seconds
    if 's' in remaining:
        # Make sure this is not milliseconds
        if not remaining.endswith('ms'):
            parts = remaining.split('s')
            seconds = int(parts[0])
            remaining = parts[1]
    
    # Create timedelta
    return datetime.timedelta(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds
    )


def get_date_range(start_date: Union[str, datetime.datetime], 
                 end_date: Union[str, datetime.datetime]) -> List[datetime.datetime]:
    """
    Get a list of dates in the range [start_date, end_date], inclusive.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        List of dates in the range
    """
    # Parse dates if they are strings
    if isinstance(start_date, str):
        start_date = parse_timestamp(start_date).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    else:
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    if isinstance(end_date, str):
        end_date = parse_timestamp(end_date).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    else:
        end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Ensure start_date <= end_date
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    
    # Generate date range
    dates = []
    current_date = start_date
    
    while current_date <= end_date:
        dates.append(current_date)
        current_date += datetime.timedelta(days=1)
    
    return dates


def get_time_ago(timestamp: Union[int, float, str, datetime.datetime],
               current_time: Optional[Union[int, float, str, datetime.datetime]] = None) -> str:
    """
    Get a human-readable string representing time elapsed since timestamp.
    
    Args:
        timestamp: The reference timestamp
        current_time: The current time (defaults to now)
        
    Returns:
        Human-readable time ago string
    """
    # Parse timestamps
    dt = parse_timestamp(timestamp)
    
    if current_time is None:
        current = get_utc_now()
    else:
        current = parse_timestamp(current_time)
    
    # Calculate the time difference
    delta = current - dt
    
    # Convert to total seconds
    seconds = delta.total_seconds()
    
    # Handle future dates
    if seconds < 0:
        return "in the future"
    
    # Convert to appropriate unit
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)} minutes ago"
    
    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)} hours ago"
    
    days = hours / 24
    if days < 30:
        return f"{int(days)} days ago"
    
    months = days / 30
    if months < 12:
        return f"{int(months)} months ago"
    
    years = days / 365
    return f"{int(years)} years ago" 