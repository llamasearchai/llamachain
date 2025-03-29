#!/usr/bin/env python
"""
Main entry point for the LlamaChain application.

This module handles the command-line execution of the LlamaChain platform,
allowing users to run the API server, background worker, or CLI.
"""

import asyncio
import argparse
import sys
from typing import List, Optional

# Import needed components
from llamachain.api.app import start_api_server
from llamachain.worker.main import run_worker
from llamachain.cli.main import run_cli
from llamachain.log import get_logger

# Set up logger
logger = get_logger("llamachain.main")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LlamaChain Platform - Blockchain Analytics and Security"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # API server command
    api_parser = subparsers.add_parser("api", help="Run the API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Worker command
    worker_parser = subparsers.add_parser("worker", help="Run the background worker")
    worker_parser.add_argument("--concurrency", type=int, default=10, 
                               help="Maximum number of concurrent tasks")
    
    # CLI command
    cli_parser = subparsers.add_parser("cli", help="Run CLI commands")
    cli_parser.add_argument("cli_args", nargs="*", help="CLI arguments to pass")
    
    return parser.parse_args(args)


async def main_async(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the LlamaChain application.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    parsed_args = parse_args(args)
    
    if parsed_args.command == "api":
        # Run API server
        logger.info(f"Starting API server on {parsed_args.host}:{parsed_args.port}")
        return await start_api_server(
            host=parsed_args.host,
            port=parsed_args.port,
            reload=parsed_args.reload
        )
    
    elif parsed_args.command == "worker":
        # Run worker
        logger.info(f"Starting background worker with concurrency {parsed_args.concurrency}")
        return await run_worker(concurrency=parsed_args.concurrency)
    
    elif parsed_args.command == "cli":
        # Run CLI command
        return await run_cli(parsed_args.cli_args)
    
    else:
        # No command specified, show help
        parse_args(["--help"])
        return 1


def main() -> int:
    """
    Run the main async function.
    
    Returns:
        Exit code
    """
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        return 0
    except Exception as e:
        logger.error(f"Application failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 