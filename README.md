# LlamaChain

A comprehensive blockchain analytics and security platform for on-chain data analysis and smart contract security auditing.

![LlamaChain Banner](assets/banner.png)

## 🌟 Features

- **Blockchain Data Access**: Connect to multiple blockchains (Ethereum, Solana) and access on-chain data.
- **Analytics Dashboard**: View and analyze blockchain data through interactive visualizations.
- **Security Auditing**: Analyze smart contracts for vulnerabilities and security issues.
- **Monitoring & Alerts**: Monitor addresses, contracts, and transactions with real-time alerts.
- **Background Workers**: Process blockchain data asynchronously for analytics and monitoring.
- **API**: Access blockchain data and analytics through a RESTful API.
- **CLI**: Command-line interface for interacting with the platform.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL (optional if using Docker)
- Docker & Docker Compose (optional)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/llamachain.git
cd llamachain
```

2. Set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment variables:

Copy the example `.env` file and modify as needed:

```bash
cp .env.example .env
```

### Using the Startup Script

LlamaChain includes a convenient startup script to manage the application:

```bash
# Make the script executable
chmod +x llamachain-script.sh

# Start all services
./llamachain-script.sh start

# Start only the API
./llamachain-script.sh api

# Start only the worker
./llamachain-script.sh worker

# Run CLI commands
./llamachain-script.sh cli blockchain list

# Get help
./llamachain-script.sh --help

# Use Docker if available
./llamachain-script.sh --docker start
```

### Running with Docker

1. Build and start services:

```bash
docker-compose up -d
```

2. Access the API at [http://localhost:8000](http://localhost:8000)

3. Run CLI commands:

```bash
docker-compose run --rm api python -m llamachain cli <command>
```

## 🔧 Usage

### API Examples

The API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs) when running the server.

#### Example Endpoints:

- `GET /dashboard/summary`: Get a summary of blockchain statistics
- `GET /dashboard/network/stats/{chain}`: Get statistics for a specific blockchain
- `GET /dashboard/transactions/recent/{chain}`: Get recent transactions
- `GET /dashboard/security/alerts`: Get security alerts

### CLI Examples

```bash
# List available blockchains
python -m llamachain cli blockchain list

# Get blockchain information
python -m llamachain cli blockchain info ethereum

# Get block information
python -m llamachain cli blockchain block ethereum 12345678

# Get transaction information
python -m llamachain cli blockchain tx ethereum 0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef

# Run security analysis
python -m llamachain cli security audit ethereum 0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef

# View configuration
python -m llamachain cli config list
```

## 📊 Dashboard

The dashboard provides visualizations and analytics for blockchain data:

- Network statistics
- Transaction volume
- Gas prices
- Security alerts
- Address analytics
- Contract analysis

## 🔒 Security Features

- Smart contract vulnerability detection
- Transaction anomaly detection
- Security alerts and notifications
- Address and contract risk scoring

## 📁 Project Structure

```
llamachain/
├── analytics/            # Analytics and data processing
├── api/                  # API server and endpoints
│   ├── app.py            # FastAPI application
│   └── endpoints/        # API route handlers
├── blockchain/           # Blockchain connectors
│   ├── base.py           # Base blockchain interface
│   ├── ethereum/         # Ethereum implementation
│   └── solana/           # Solana implementation
├── cli/                  # Command-line interface
├── db/                   # Database models and session
├── worker/               # Background worker processes
├── __main__.py           # Entry point for the application
├── config.py             # Configuration settings
└── log.py                # Logging utilities
```

## 💻 Development

### Running Tests

```bash
pytest
```

### Linting

```bash
flake8 llamachain
black llamachain
```

### Database Migrations

```bash
alembic revision --autogenerate -m "description"
alembic upgrade head
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

## 📢 Acknowledgements

- [Web3.py](https://github.com/ethereum/web3.py)
- [Solana.py](https://github.com/michaelhly/solana-py)
- [FastAPI](https://fastapi.tiangolo.com/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [Plotly](https://plotly.com/) 