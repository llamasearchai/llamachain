"""
Worker module for the LlamaChain platform.

This module provides background processing capabilities for tasks such as
blockchain monitoring, data processing, and analytics.
"""

import asyncio
import signal
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

from llamachain.db.session import get_db
from llamachain.log import get_logger

# Setup logger
logger = get_logger("llamachain.worker")


class WorkerTask:
    """
    A task that can be run by the worker.
    
    Attributes:
        name: Name of the task
        func: Async function to run
        interval_seconds: Time between task runs in seconds
        last_run: Timestamp of the last run
        running: Whether the task is currently running
        should_cancel: Whether the task should be cancelled
    """
    
    def __init__(
        self, 
        name: str, 
        func: Callable, 
        interval_seconds: int
    ):
        """
        Initialize a worker task.
        
        Args:
            name: Name of the task
            func: Async function to run
            interval_seconds: Time between task runs in seconds
        """
        self.name = name
        self.func = func
        self.interval_seconds = interval_seconds
        self.last_run: Optional[datetime] = None
        self.running = False
        self.should_cancel = False
    
    async def run(self) -> None:
        """Run the task."""
        if self.running:
            logger.warning(f"Task {self.name} is already running, skipping")
            return
        
        self.running = True
        self.last_run = datetime.now()
        logger.info(f"Running task {self.name}")
        
        try:
            await self.func()
            logger.info(f"Task {self.name} completed successfully")
        except Exception as e:
            logger.error(f"Task {self.name} failed: {e}")
        finally:
            self.running = False
    
    def should_run(self) -> bool:
        """
        Check if the task should run based on its interval.
        
        Returns:
            True if the task should run, False otherwise
        """
        if self.running or self.should_cancel:
            return False
        
        if self.last_run is None:
            return True
        
        next_run = self.last_run + timedelta(seconds=self.interval_seconds)
        return datetime.now() >= next_run


class Worker:
    """
    Background worker that runs tasks periodically.
    
    Attributes:
        tasks: Dictionary of registered tasks
        concurrency: Maximum number of concurrent tasks
        running_tasks: Set of running task names
        running: Whether the worker is running
    """
    
    def __init__(self, concurrency: int = 10):
        """
        Initialize the worker.
        
        Args:
            concurrency: Maximum number of concurrent tasks
        """
        self.tasks: Dict[str, WorkerTask] = {}
        self.concurrency = concurrency
        self.running_tasks: Set[str] = set()
        self.running = False
    
    def register_task(
        self, 
        name: str, 
        func: Callable, 
        interval_seconds: int
    ) -> None:
        """
        Register a task with the worker.
        
        Args:
            name: Name of the task
            func: Async function to run
            interval_seconds: Time between task runs in seconds
        """
        self.tasks[name] = WorkerTask(name, func, interval_seconds)
        logger.info(f"Registered task {name} with interval {interval_seconds}s")
    
    async def run(self) -> int:
        """
        Run the worker until stopped.
        
        Returns:
            Exit code
        """
        self.running = True
        logger.info(f"Starting worker with {len(self.tasks)} registered tasks")
        
        # Register signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_stop_signal)
        
        while self.running:
            try:
                # Run eligible tasks up to concurrency limit
                available_slots = self.concurrency - len(self.running_tasks)
                
                if available_slots > 0:
                    for name, task in self.tasks.items():
                        if name not in self.running_tasks and task.should_run() and available_slots > 0:
                            self.running_tasks.add(name)
                            asyncio.create_task(self._run_task(name))
                            available_slots -= 1
                
                # Sleep briefly to prevent high CPU usage
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
                self.running = False
                return 1
        
        logger.info("Worker stopped")
        return 0
    
    def stop(self) -> None:
        """Stop the worker."""
        if self.running:
            logger.info("Stopping worker...")
            self.running = False
            
            # Cancel all tasks
            for task in self.tasks.values():
                task.should_cancel = True
    
    def _handle_stop_signal(self) -> None:
        """Handle stop signals (SIGINT, SIGTERM)."""
        logger.info("Received stop signal")
        self.stop()
    
    async def _run_task(self, name: str) -> None:
        """
        Run a task and handle cleanup.
        
        Args:
            name: Name of the task to run
        """
        try:
            await self.tasks[name].run()
        finally:
            if name in self.running_tasks:
                self.running_tasks.remove(name)


# Define some example worker tasks

async def monitor_blockchain_activity() -> None:
    """
    Monitor blockchain activity for new blocks and transactions.
    
    This task periodically checks for new blocks and transactions on supported blockchains
    and stores the data in the database.
    """
    logger.info("Monitoring blockchain activity")
    
    try:
        # This would normally connect to blockchain nodes and fetch data
        # Example: Get latest blocks from Ethereum and Solana
        
        # Get database session
        async for db in get_db():
            # Store the data in the database
            # This is just a placeholder
            pass
            
        logger.info("Blockchain monitoring completed")
    except Exception as e:
        logger.error(f"Error monitoring blockchain activity: {e}")


async def update_price_data() -> None:
    """
    Update price data for tracked tokens.
    
    This task periodically fetches price data for tracked tokens from price APIs
    and stores the data in the database.
    """
    logger.info("Updating price data")
    
    try:
        # This would normally fetch price data from APIs
        # Example: Get token prices from CoinGecko
        
        # Get database session
        async for db in get_db():
            # Store the data in the database
            # This is just a placeholder
            pass
            
        logger.info("Price data update completed")
    except Exception as e:
        logger.error(f"Error updating price data: {e}")


async def run_analytics_tasks() -> None:
    """
    Run analytics tasks.
    
    This task periodically runs analytics tasks to derive insights from
    blockchain data and stores the results in the database.
    """
    logger.info("Running analytics tasks")
    
    try:
        # This would normally run analytics algorithms
        # Example: Analyze transaction patterns, detect anomalies
        
        # Get database session
        async for db in get_db():
            # Store the results in the database
            # This is just a placeholder
            pass
            
        logger.info("Analytics tasks completed")
    except Exception as e:
        logger.error(f"Error running analytics tasks: {e}")


async def check_security_alerts() -> None:
    """
    Check for security alerts.
    
    This task periodically checks for security vulnerabilities and suspicious
    activity on monitored contracts and addresses.
    """
    logger.info("Checking security alerts")
    
    try:
        # This would normally run security checks
        # Example: Check for reentrancy, overflow, etc.
        
        # Get database session
        async for db in get_db():
            # Store alerts in the database
            # This is just a placeholder
            pass
            
        logger.info("Security checks completed")
    except Exception as e:
        logger.error(f"Error checking security alerts: {e}")


async def run_worker(concurrency: int = 10) -> int:
    """
    Initialize and run the worker.
    
    Args:
        concurrency: Maximum number of concurrent tasks
        
    Returns:
        Exit code
    """
    worker = Worker(concurrency=concurrency)
    
    # Register tasks
    worker.register_task("blockchain_monitor", monitor_blockchain_activity, 60)  # Every minute
    worker.register_task("price_update", update_price_data, 300)  # Every 5 minutes
    worker.register_task("analytics", run_analytics_tasks, 3600)  # Every hour
    worker.register_task("security_check", check_security_alerts, 1800)  # Every 30 minutes
    
    # Run the worker
    return await worker.run() 