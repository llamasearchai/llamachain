"""
LlamaChain - Blockchain Analytics and Security Platform

A comprehensive platform for blockchain data analytics, security analysis,
and smart contract auditing.
"""

__version__ = "0.1.0"
__author__ = "LlamaChain Team"

# Import core modules for easy access
from llamachain import api
from llamachain import blockchain
from llamachain import db
from llamachain import analytics
from llamachain import worker
from llamachain import cli

# Make version available
from .__version__ import VERSION as __version__

# Import commonly used modules for easier access
from llamachain.config import settings
from llamachain.log import get_logger, setup_logging 