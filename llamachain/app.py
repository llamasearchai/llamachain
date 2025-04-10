"""
Main application module for the LlamaChain platform.

This module initializes and runs the FastAPI application.
"""

import logging
import os
from typing import Dict, Any

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from llamachain.api.routes import router as api_router
from llamachain.core.config import settings
from llamachain.db import init_db
from llamachain.log import get_logger, setup_logging
from llamachain.nlp.processor import NLPProcessor

# Setup logging
setup_logging()
logger = get_logger("llamachain.app")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Initialize NLP processor
nlp_processor = NLPProcessor()

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("Mounted static files directory")
except Exception as e:
    logger.warning(f"Could not mount static files directory: {str(e)}")

# Setup templates
try:
    templates = Jinja2Templates(directory="templates")
    logger.info("Initialized Jinja2 templates")
except Exception as e:
    logger.warning(f"Could not initialize Jinja2 templates: {str(e)}")
    templates = None


@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    logger.info("Starting LlamaChain application")
    
    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run shutdown tasks."""
    logger.info("Shutting down LlamaChain application")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/info")
async def info():
    """Information endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
    }


@app.post("/query")
async def process_query(request: Dict[str, Any]):
    """
    Process a natural language query.
    
    Args:
        request: Request data containing the query
        
    Returns:
        Processed query information
    """
    try:
        query = request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Process the query
        processed_query = await nlp_processor.process_query(query)
        
        # Generate response
        response = await nlp_processor.generate_response(processed_query)
        
        # Translate to structured query
        structured_query = await nlp_processor.translate_to_query(processed_query)
        
        return {
            "query": query,
            "intent": processed_query["intent"],
            "entities": processed_query["entities"],
            "response": response,
            "structured_query": structured_query,
        }
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"},
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "llamachain.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    ) 