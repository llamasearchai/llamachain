"""
API server for the LlamaChain platform.

This module defines the FastAPI application and server for the LlamaChain API.
"""

import asyncio
from typing import Optional, Dict, Any
import time

import uvicorn
from fastapi import FastAPI, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from llamachain.config import settings
from llamachain.db.session import init_db
from llamachain.log import get_logger
from llamachain.api.endpoints.dashboard import router as dashboard_router

# Import API routers
from llamachain.api.endpoints.blockchain import router as blockchain_router

# Setup logger
logger = get_logger("llamachain.api")

# Create FastAPI app
app = FastAPI(
    title="LlamaChain API",
    description="API for the LlamaChain platform",
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip middleware for compression
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next) -> Response:
    """
    Log request information and timing.
    
    Args:
        request: The incoming request
        call_next: The next middleware or route handler
        
    Returns:
        The response
    """
    start_time = time.time()
    path = request.url.path
    query = str(request.url.query)
    query_text = f"?{query}" if query else ""
    
    logger.info(f"Request: {request.method} {path}{query_text}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logger.info(
            f"Response: {request.method} {path} - Status: {response.status_code} - "
            f"Time: {process_time:.4f}s"
        )
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        logger.error(f"Request error: {e}")
        process_time = time.time() - start_time
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"},
            headers={"X-Process-Time": str(process_time)},
        )


# Add startup and shutdown events
@app.on_event("startup")
async def startup_db_client() -> None:
    """Initialize database on startup."""
    logger.info("Initializing database connection")
    await init_db()
    logger.info("Database initialized")


@app.on_event("shutdown")
async def shutdown_db_client() -> None:
    """Close database connections on shutdown."""
    logger.info("Closing database connections")
    # Any cleanup code would go here


# Root endpoint
@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint that returns API information.
    
    Returns:
        Dict with API information
    """
    return {
        "name": "LlamaChain API",
        "version": settings.API_VERSION,
        "description": "Blockchain Analytics and Security Platform API",
        "docs": "/docs",
    }


# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns:
        Dict with health status
    """
    return {
        "status": "ok",
        "version": settings.API_VERSION,
        "timestamp": time.time(),
    }


# Register routers
app.include_router(dashboard_router, prefix="/dashboard", tags=["dashboard"])
app.include_router(blockchain_router, prefix="/api/blockchain", tags=["blockchain"])


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


async def start_api_server(
    host: str = settings.API_HOST,
    port: int = settings.API_PORT,
    reload: bool = False
) -> int:
    """
    Start the API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Whether to enable auto-reload
        
    Returns:
        Exit code
    """
    config = uvicorn.Config(
        "llamachain.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
    
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
        return 0
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        return 1 