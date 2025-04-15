"""
API routes for the LlamaChain platform.
"""

from fastapi import APIRouter

from llamachain.api.routes import analytics, blockchain, nlp, security

# Create main router
router = APIRouter()

# Include routers
router.include_router(blockchain.router)
router.include_router(security.router)
router.include_router(analytics.router)
router.include_router(nlp.router)
