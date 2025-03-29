from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated, List

from llamachain.api.endpoints import (
    blockchain_router,
    analysis_router,
    security_router,
    ai_router,
    dashboard_router,
)
from llamachain.utils.config import settings
from llamachain.db.session import engine, SessionLocal
from llamachain.db.base import Base
from llamachain.blockchain.ethereum.chain import EthereumChain
from llamachain.blockchain.solana.chain import SolanaChain

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Lifespan for application startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create database tables, initialize connections
    logger.info("Starting up LlamaChain API server...")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Initialize blockchain connections
    app.state.eth_chain = EthereumChain(
        rpc_url=settings.ETH_RPC_URL,
        ws_url=settings.ETH_WSS_URL
    )
    await app.state.eth_chain.connect()
    
    app.state.sol_chain = SolanaChain(
        rpc_url=settings.SOL_RPC_URL
    )
    await app.state.sol_chain.connect()
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown: Close connections and cleanup resources
    logger.info("Shutting down LlamaChain API server...")
    
    await app.state.eth_chain.disconnect()
    await app.state.sol_chain.disconnect()
    
    await engine.dispose()
    
    logger.info("Application shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="LlamaChain API",
    description="Advanced Blockchain Intelligence Platform API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url=None,  # We'll use custom docs
    redoc_url=None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")

# Custom docs with better styling
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API Documentation",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url="/static/favicon.png",
    )

# Application health checks
@app.get("/health")
async def health_check():
    """Health check endpoint for the API."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": app.version,
    }

@app.get("/health/blockchain")
async def blockchain_health():
    """Check blockchain connection health."""
    eth_status = await app.state.eth_chain.is_connected()
    sol_status = await app.state.sol_chain.is_connected()
    
    return {
        "ethereum": {
            "connected": eth_status,
            "chain_id": await app.state.eth_chain.get_chain_id() if eth_status else None,
            "latest_block": await app.state.eth_chain.get_latest_block_number() if eth_status else None,
        },
        "solana": {
            "connected": sol_status,
            "slot": await app.state.sol_chain.get_latest_slot() if sol_status else None,
        }
    }

# Register routers
app.include_router(blockchain_router, prefix="/api/v1/blockchain", tags=["Blockchain"])
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["Analysis"])
app.include_router(security_router, prefix="/api/v1/security", tags=["Security"])
app.include_router(ai_router, prefix="/api/v1/ai", tags=["AI"])
app.include_router(dashboard_router, prefix="/api/v1/dashboard", tags=["Dashboard"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "LlamaChain API",
        "version": app.version,
        "description": "Advanced Blockchain Intelligence Platform API",
        "docs": "/docs",
    } 