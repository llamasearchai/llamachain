"""
API endpoints package for the LlamaChain platform.

This package contains the API endpoint modules for different resource types.
"""

# Import endpoints modules
from llamachain.api.endpoints import dashboard

from llamachain.api.endpoints.blockchain import router as blockchain_router
from llamachain.api.endpoints.analysis import router as analysis_router
from llamachain.api.endpoints.security import router as security_router
from llamachain.api.endpoints.ai import router as ai_router
from llamachain.api.endpoints.dashboard import router as dashboard_router

__all__ = [
    "blockchain_router",
    "analysis_router",
    "security_router",
    "ai_router",
    "dashboard_router",
] 