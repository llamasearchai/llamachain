# LlamaChain Blockchain Module

This directory contains the blockchain integration components for the LlamaChain platform.

## Structure

- `base.py`: Abstract base class for blockchain implementations
- `registry.py`: Registry for managing blockchain connections
- `ethereum/`: Ethereum blockchain implementation
  - `chain.py`: Ethereum chain implementation
  - `utils.py`: Ethereum-specific utilities
- `solana/`: Solana blockchain implementation
  - `chain.py`: Solana chain implementation
  - `utils.py`: Solana-specific utilities

## Usage

```python
from llamachain.blockchain import BlockchainRegistry

# Get a blockchain instance
registry = BlockchainRegistry()
ethereum = await registry.get_chain("ethereum")
solana = await registry.get_chain("solana")

# Connect to the blockchain
await ethereum.connect()

# Get the latest block
latest_block = await ethereum.get_latest_block()

# Get a transaction
tx = await ethereum.get_transaction("0x...")

# Get an account balance
balance = await ethereum.get_balance("0x...")

# Disconnect when done
await ethereum.disconnect()
```

## Adding a New Blockchain

To add support for a new blockchain:

1. Create a new directory for the blockchain (e.g., `llamachain/blockchain/polygon/`)
2. Create a chain implementation that inherits from `BlockchainBase`
3. Implement all required methods
4. Register the chain in `registry.py`

Example:

```python
from llamachain.blockchain.base import BlockchainBase

class PolygonChain(BlockchainBase):
    """Polygon blockchain implementation."""
    
    def __init__(self, rpc_url=None, ws_url=None):
        super().__init__()
        self.rpc_url = rpc_url or "https://polygon-rpc.com"
        self.ws_url = ws_url
        self.web3 = None
        
    async def connect(self):
        # Implementation
        pass
        
    async def disconnect(self):
        # Implementation
        pass
        
    async def is_connected(self):
        # Implementation
        pass
        
    # Implement other required methods...
```

Then register it in the registry:

```python
from llamachain.blockchain.polygon.chain import PolygonChain

registry = BlockchainRegistry()
registry.register_chain_class("polygon", PolygonChain)
``` 