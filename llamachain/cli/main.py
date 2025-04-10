"""
Command-line interface (CLI) for the LlamaChain platform.

This module provides a command-line interface for interacting with the LlamaChain platform,
allowing users to perform various operations like blockchain data access, security checks,
analytics, and configuration management.
"""

import argparse
import asyncio
import sys
from typing import Any, Dict, List, Optional, Tuple, Type

from llamachain.blockchain import BlockchainRegistry
from llamachain.config import settings
from llamachain.log import get_logger
from llamachain.blockchain.base import BlockchainBase
from llamachain.blockchain.ethereum import EthereumChain
from llamachain.blockchain.solana import SolanaChain
from llamachain.security.audit import SmartContractAuditor
from llamachain.security.rules import VulnerabilityType
from llamachain.analytics.dashboard import get_network_summary, get_recent_transactions
from llamachain.nlp.processor import NLPProcessor
from llamachain.core.config import get_config, set_config

# Setup logger
logger = get_logger("llamachain.cli")


class CLICommand:
    """
    Base class for CLI commands.
    
    Attributes:
        name: Command name
        description: Command description
    """
    
    name = "command"
    description = "Base command"
    
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """
        Add command-specific arguments to the parser.
        
        Args:
            parser: Argument parser instance
        """
        pass
    
    @classmethod
    async def run(cls, args: argparse.Namespace) -> int:
        """
        Run the command.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            Exit code
        """
        raise NotImplementedError("Command must implement run method")


class BlockchainCommands(CLICommand):
    """Commands for blockchain data access."""
    
    name = "blockchain"
    description = "Access blockchain data"
    
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add blockchain command arguments."""
        subparsers = parser.add_subparsers(dest="blockchain_command", help="Blockchain command")
        
        # List supported blockchains
        list_parser = subparsers.add_parser("list", help="List available blockchains")
        
        # Get blockchain info
        info_parser = subparsers.add_parser("info", help="Get blockchain information")
        info_parser.add_argument("chain", help="Blockchain name (ethereum, solana)")
        
        # Get block information
        block_parser = subparsers.add_parser("block", help="Get block information")
        block_parser.add_argument("chain", help="Blockchain name (ethereum, solana)")
        block_parser.add_argument("block", help="Block number or hash")
        
        # Get transaction information
        tx_parser = subparsers.add_parser("tx", help="Get transaction information")
        tx_parser.add_argument("chain", help="Blockchain name (ethereum, solana)")
        tx_parser.add_argument("tx_hash", help="Transaction hash")
        
        # Get address information
        address_parser = subparsers.add_parser("address", help="Get address information")
        address_parser.add_argument("chain", help="Blockchain name (ethereum, solana)")
        address_parser.add_argument("address", help="Blockchain address")
        
        # Get gas price
        gas_parser = subparsers.add_parser("gas", help="Get gas price")
        gas_parser.add_argument("chain", help="Blockchain name (ethereum, solana)")
    
    @classmethod
    async def run(cls, args: argparse.Namespace) -> int:
        """Run blockchain command."""
        if not hasattr(args, "blockchain_command") or not args.blockchain_command:
            print(f"Error: No blockchain command specified")
            return 1
        
        registry = BlockchainRegistry()
        
        if args.blockchain_command == "list":
            # List available blockchains
            chains = registry.get_available_chains()
            print("Available blockchains:")
            for chain in chains:
                print(f"  - {chain}")
            return 0
            
        elif args.blockchain_command == "info":
            # Get blockchain information
            try:
                chain = await registry.get_chain(args.chain)
                if not await chain.is_connected():
                    await chain.connect()
                
                chain_id = await chain.get_chain_id()
                chain_name = await chain.get_chain_name()
                latest_block = await chain.get_latest_block()
                
                print(f"Chain: {chain_name}")
                print(f"Chain ID: {chain_id}")
                print(f"Latest block: {latest_block['number']}")
                print(f"Block time: {latest_block['timestamp']}")
                print(f"Transaction count: {len(latest_block.get('transactions', []))}")
                
                return 0
            except Exception as e:
                print(f"Error getting blockchain info: {e}")
                return 1
                
        elif args.blockchain_command == "block":
            # Get block information
            try:
                chain = await registry.get_chain(args.chain)
                if not await chain.is_connected():
                    await chain.connect()
                
                block = await chain.get_block(args.block)
                
                print(f"Block: {block['number']}")
                print(f"Hash: {block['hash']}")
                print(f"Timestamp: {block['timestamp']}")
                print(f"Parent hash: {block.get('parent_hash', 'N/A')}")
                print(f"Transactions: {len(block.get('transactions', []))}")
                
                return 0
            except Exception as e:
                print(f"Error getting block info: {e}")
                return 1
                
        elif args.blockchain_command == "tx":
            # Get transaction information
            try:
                chain = await registry.get_chain(args.chain)
                if not await chain.is_connected():
                    await chain.connect()
                
                tx = await chain.get_transaction(args.tx_hash)
                receipt = await chain.get_transaction_receipt(args.tx_hash)
                
                print(f"Transaction: {tx['hash']}")
                print(f"Block: {tx.get('block_number', 'Pending')}")
                print(f"From: {tx['from']}")
                print(f"To: {tx.get('to', 'Contract creation')}")
                print(f"Value: {tx.get('value', 0)}")
                print(f"Gas price: {tx.get('gas_price', 'N/A')}")
                print(f"Gas used: {receipt.get('gas_used', 'N/A')}")
                print(f"Status: {'Success' if receipt.get('status') == 1 else 'Failed'}")
                
                return 0
            except Exception as e:
                print(f"Error getting transaction info: {e}")
                return 1
                
        elif args.blockchain_command == "address":
            # Get address information
            try:
                chain = await registry.get_chain(args.chain)
                if not await chain.is_connected():
                    await chain.connect()
                
                balance = await chain.get_balance(args.address)
                code = await chain.get_contract_code(args.address)
                is_contract = code != "0x" and code is not None
                
                print(f"Address: {args.address}")
                print(f"Balance: {balance}")
                print(f"Is contract: {is_contract}")
                
                if is_contract:
                    print(f"Code size: {len(code or '0x') // 2} bytes")
                
                return 0
            except Exception as e:
                print(f"Error getting address info: {e}")
                return 1
                
        elif args.blockchain_command == "gas":
            # Get gas price
            try:
                chain = await registry.get_chain(args.chain)
                if not await chain.is_connected():
                    await chain.connect()
                
                gas_price = await chain.get_gas_price()
                
                print(f"Current gas price: {gas_price}")
                
                return 0
            except Exception as e:
                print(f"Error getting gas price: {e}")
                return 1
        
        else:
            print(f"Error: Unknown blockchain command: {args.blockchain_command}")
            return 1


class SecurityCommands(CLICommand):
    """Commands for security analysis."""
    
    name = "security"
    description = "Security analysis tools"
    
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add security command arguments."""
        subparsers = parser.add_subparsers(dest="security_command", help="Security command")
        
        # Audit contract
        audit_parser = subparsers.add_parser("audit", help="Audit a smart contract")
        audit_parser.add_argument("chain", help="Blockchain name (ethereum, solana)")
        audit_parser.add_argument("address", help="Contract address")
        
        # List vulnerability types
        vuln_parser = subparsers.add_parser("vulnerabilities", help="List vulnerability types")
    
    @classmethod
    async def run(cls, args: argparse.Namespace) -> int:
        """Run security command."""
        if not hasattr(args, "security_command") or not args.security_command:
            print(f"Error: No security command specified")
            return 1
        
        if args.security_command == "audit":
            # Audit contract
            print(f"Auditing contract {args.address} on {args.chain}...")
            print("Note: This feature is not yet implemented")
            return 0
            
        elif args.security_command == "vulnerabilities":
            # List vulnerability types
            print("Vulnerability types:")
            print("  - Reentrancy")
            print("  - Integer Overflow/Underflow")
            print("  - Unchecked External Calls")
            print("  - Front-Running")
            print("  - Timestamp Dependence")
            print("  - Access Control Issues")
            print("  - Denial of Service")
            print("  - Logic Errors")
            return 0
            
        else:
            print(f"Error: Unknown security command: {args.security_command}")
            return 1


class AnalyticsCommands(CLICommand):
    """Commands for analytics."""
    
    name = "analytics"
    description = "Analytics tools"
    
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add analytics command arguments."""
        subparsers = parser.add_subparsers(dest="analytics_command", help="Analytics command")
        
        # Token price analytics
        token_parser = subparsers.add_parser("token", help="Token price analytics")
        token_parser.add_argument("symbol", help="Token symbol")
        token_parser.add_argument("--days", type=int, default=7, help="Number of days (default: 7)")
        
        # Address analytics
        address_parser = subparsers.add_parser("address", help="Address analytics")
        address_parser.add_argument("chain", help="Blockchain name (ethereum, solana)")
        address_parser.add_argument("address", help="Blockchain address")
        
        # Gas price analytics
        gas_parser = subparsers.add_parser("gas", help="Gas price analytics")
        gas_parser.add_argument("chain", help="Blockchain name (ethereum, solana)")
        gas_parser.add_argument("--days", type=int, default=7, help="Number of days (default: 7)")
    
    @classmethod
    async def run(cls, args: argparse.Namespace) -> int:
        """Run analytics command."""
        if not hasattr(args, "analytics_command") or not args.analytics_command:
            print(f"Error: No analytics command specified")
            return 1
        
        if args.analytics_command == "token":
            # Token price analytics
            print(f"Token price analytics for {args.symbol} over {args.days} days")
            print("Note: This feature is not yet implemented")
            return 0
            
        elif args.analytics_command == "address":
            # Address analytics
            print(f"Address analytics for {args.address} on {args.chain}")
            print("Note: This feature is not yet implemented")
            return 0
            
        elif args.analytics_command == "gas":
            # Gas price analytics
            print(f"Gas price analytics for {args.chain} over {args.days} days")
            print("Note: This feature is not yet implemented")
            return 0
            
        else:
            print(f"Error: Unknown analytics command: {args.analytics_command}")
            return 1


class NLPCommands(CLICommand):
    """Commands for natural language processing."""
    
    name = "nlp"
    description = "Natural language processing"
    
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add NLP command arguments."""
        subparsers = parser.add_subparsers(dest="nlp_command", help="NLP command")
        
        # Query command
        query_parser = subparsers.add_parser("query", help="Process a natural language query")
        query_parser.add_argument("--text", "-t", required=True, help="Query text")
        query_parser.add_argument("--response", "-r", action="store_true", help="Generate response")
        query_parser.add_argument("--structured", "-s", action="store_true", help="Translate to structured query")
        query_parser.add_argument("--execute", "-e", action="store_true", help="Execute the query if possible")
    
    @classmethod
    async def run(cls, args: argparse.Namespace) -> int:
        """Run the NLP command."""
        if not hasattr(args, "nlp_command") or not args.nlp_command:
            print(f"Error: No NLP command specified")
            return 1
        
        if args.nlp_command == "query":
            await cls._process_query(args)
        else:
            print(f"Error: Unknown NLP command: {args.nlp_command}")
            return 1
    
    @classmethod
    async def _process_query(cls, args: argparse.Namespace) -> int:
        """Process a natural language query."""
        try:
            # Create NLP processor
            processor = NLPProcessor()
            
            # Process the query
            processed_query = await processor.process_query(args.text)
            
            # Print query information
            print(f"Query: {processed_query['original_query']}")
            print(f"Intent: {processed_query['intent']['type']} ({processed_query['intent']['description']})")
            
            # Print entities
            if processed_query['entities']:
                print("\nEntities:")
                for entity in processed_query['entities']:
                    print(f"- {entity['type']}: {entity['value']} (confidence: {entity['confidence']:.2f})")
            else:
                print("\nNo entities found.")
            
            # Generate response if requested
            if args.response:
                response = await processor.generate_response(processed_query)
                print(f"\nResponse: {response}")
            
            # Translate to structured query if requested
            if args.structured:
                structured_query = await processor.translate_to_query(processed_query)
                print("\nStructured Query:")
                print(f"Type: {structured_query['type']}")
                print("Parameters:")
                for key, value in structured_query['params'].items():
                    print(f"- {key}: {value}")
            
            # Execute the query if requested
            if args.execute:
                print("\nQuery execution is not yet implemented.")
                print("This feature will be available in a future release.")
        
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return 1


class ConfigCommands(CLICommand):
    """Commands for configuration."""
    
    name = "config"
    description = "Configuration management"
    
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add configuration command arguments."""
        subparsers = parser.add_subparsers(dest="config_command", help="Config command")
        
        # List configuration
        list_parser = subparsers.add_parser("list", help="List configuration settings")
        
        # Set configuration
        set_parser = subparsers.add_parser("set", help="Set configuration setting")
        set_parser.add_argument("key", help="Setting key")
        set_parser.add_argument("value", help="Setting value")
    
    @classmethod
    async def run(cls, args: argparse.Namespace) -> int:
        """Run configuration command."""
        if not hasattr(args, "config_command") or not args.config_command:
            print(f"Error: No config command specified")
            return 1
        
        if args.config_command == "list":
            # List configuration
            print("Configuration settings:")
            for key, value in settings.dict().items():
                # Skip internal settings (those starting with _)
                if not key.startswith("_"):
                    print(f"  {key} = {value}")
            return 0
            
        elif args.config_command == "set":
            # Set configuration
            print(f"Setting {args.key} to {args.value}")
            print("Note: This feature is not yet implemented")
            print("Please edit the .env file directly to change configuration settings.")
            return 0
            
        else:
            print(f"Error: Unknown config command: {args.config_command}")
            return 1


# Map of command names to command classes
COMMANDS: Dict[str, Type[CLICommand]] = {
    "blockchain": BlockchainCommands,
    "security": SecurityCommands,
    "analytics": AnalyticsCommands,
    "nlp": NLPCommands,
    "config": ConfigCommands,
}


def print_help() -> None:
    """Print command help."""
    print("LlamaChain CLI")
    print("Available commands:")
    for name, cmd in COMMANDS.items():
        print(f"  {name:<15} - {cmd.description}")
    print("\nUse '<command> --help' for more information about a command.")


async def run_command(args: List[str]) -> int:
    """
    Run a CLI command.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    if not args:
        print_help()
        return 0
    
    command_name = args[0]
    
    if command_name in ("-h", "--help"):
        print_help()
        return 0
    
    if command_name not in COMMANDS:
        print(f"Error: Unknown command: {command_name}")
        print_help()
        return 1
    
    command_class = COMMANDS[command_name]
    
    # Create parser for this command
    parser = argparse.ArgumentParser(prog=f"llamachain {command_name}")
    command_class.add_arguments(parser)
    
    # Parse arguments for this command
    parsed_args = parser.parse_args(args[1:])
    
    # Run command
    try:
        return await command_class.run(parsed_args)
    except Exception as e:
        logger.error(f"Error running command {command_name}: {e}")
        print(f"Error: {e}")
        return 1


async def run_cli(args: Optional[List[str]] = None) -> int:
    """
    Entry point for the CLI.
    
    Args:
        args: Command-line arguments, defaults to sys.argv[1:]
        
    Returns:
        Exit code
    """
    if args is None:
        args = sys.argv[1:]
    
    return await run_command(args) 