# LlamaChain Project Structure

This document provides an overview of the LlamaChain project structure.

## Directory Structure

```
llamachain/
├── api/                 # FastAPI implementation
│   ├── endpoints/       # API endpoints
│   ├── schemas/         # Pydantic models
│   └── app.py           # Main application
├── blockchain/          # Blockchain interfaces
│   ├── ethereum/        # Ethereum integration
│   ├── solana/          # Solana integration
│   └── base.py          # Base blockchain interface
├── analytics/           # Data analytics modules
│   ├── price.py         # Price analysis
│   ├── patterns.py      # Pattern detection
│   ├── transactions.py  # Transaction analysis
│   ├── addresses.py     # Address analysis
│   └── visualizations.py # Visualization generators
├── security/            # Security analysis tools
│   ├── auditor.py       # Contract auditor
│   ├── zk.py            # Zero-knowledge verification
│   └── mev.py           # MEV protection
├── ml/                  # Machine learning models
│   ├── models/          # Model implementations
│   ├── embeddings.py    # Embedding generators
│   └── training.py      # Model training utilities
├── nlp/                 # Natural language processing
│   └── interface.py     # Natural language interface
├── db/                  # Database models and session
│   ├── models.py        # SQLAlchemy models
│   └── session.py       # Database session management
├── web/                 # Web UI and dashboard
│   ├── dashboard/       # Dashboard UI
│   └── static/          # Static assets
├── cli/                 # Command-line interface
│   └── main.py          # CLI implementation
├── worker/              # Background worker processes
│   └── main.py          # Worker implementation
├── config.py            # Configuration management
├── log.py               # Logging utilities
└── __main__.py          # Entry point
```

## Key Components

### API Layer

The API layer is built with FastAPI and provides endpoints for interacting with the LlamaChain platform. The endpoints are organized by functionality:

- `blockchain`: Blockchain data access (blocks, transactions, accounts)
- `analysis`: Analytics and pattern detection
- `security`: Smart contract security analysis
- `ai`: AI-powered predictions and classification
- `dashboard`: Data for dashboard visualizations

### Blockchain Layer

The blockchain layer provides a unified interface for interacting with different blockchain networks. The base interface is defined in `blockchain/base.py`, and implementations for specific blockchains are provided in subdirectories:

- `ethereum`: Ethereum blockchain implementation
- `solana`: Solana blockchain implementation

### Analytics Layer

The analytics layer provides tools for analyzing blockchain data, detecting patterns, and generating visualizations:

- `price.py`: Price analysis and correlation
- `patterns.py`: Pattern detection in blockchain data
- `transactions.py`: Transaction analysis
- `addresses.py`: Address analysis and classification
- `visualizations.py`: Visualization generators for dashboard

### Security Layer

The security layer provides tools for analyzing smart contracts, detecting vulnerabilities, and protecting transactions:

- `auditor.py`: Smart contract auditor
- `zk.py`: Zero-knowledge verification
- `mev.py`: MEV protection strategies

### Machine Learning Layer

The machine learning layer provides models for various tasks, including vulnerability detection, address classification, and price prediction:

- `models/`: Model implementations
- `embeddings.py`: Embedding generators for contracts and addresses
- `training.py`: Model training utilities

### Natural Language Processing Layer

The NLP layer provides a natural language interface for querying blockchain data:

- `interface.py`: Natural language interface implementation

### Database Layer

The database layer provides models and utilities for interacting with the database:

- `models.py`: SQLAlchemy models
- `session.py`: Database session management

### Web Layer

The web layer provides a dashboard UI for visualizing blockchain data:

- `dashboard/`: Dashboard UI implementation
- `static/`: Static assets (CSS, JavaScript, images)

### CLI Layer

The CLI layer provides a command-line interface for interacting with the LlamaChain platform:

- `main.py`: CLI implementation

### Worker Layer

The worker layer provides background worker processes for tasks like monitoring and analytics:

- `main.py`: Worker implementation

## Configuration and Utilities

- `config.py`: Configuration management
- `log.py`: Logging utilities
- `__main__.py`: Entry point for running the application 