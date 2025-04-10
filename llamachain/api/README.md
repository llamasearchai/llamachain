# LlamaChain API

This directory contains the FastAPI implementation for the LlamaChain platform.

## Structure

- `app.py`: Main FastAPI application
- `endpoints/`: API endpoint modules
  - `blockchain.py`: Blockchain data access endpoints
  - `analysis.py`: Analytics and pattern detection endpoints
  - `security.py`: Smart contract security analysis endpoints
  - `ai.py`: AI-powered predictions and classification endpoints
  - `dashboard.py`: Data for dashboard visualizations endpoints
- `schemas/`: Pydantic models for request/response validation

## API Documentation

When the server is running, API documentation is available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Key API Endpoints

- `/api/blockchain/*`: Blockchain data access (blocks, transactions, accounts)
- `/api/analysis/*`: Analytics and pattern detection
- `/api/security/*`: Smart contract security analysis
- `/api/ai/*`: AI-powered predictions and classification
- `/api/dashboard/*`: Data for dashboard visualizations

## Development

To add a new endpoint:

1. Create a new endpoint function in the appropriate module in `endpoints/`
2. Define request/response models in `schemas/` if needed
3. Add the endpoint to the router in the module
4. Import the router in `__init__.py` if it's a new module

Example:

```python
from fastapi import APIRouter, Depends, HTTPException
from typing import List

from llamachain.api.schemas.my_schema import MyRequestModel, MyResponseModel

router = APIRouter()

@router.post("/my-endpoint", response_model=MyResponseModel)
async def my_endpoint(request: MyRequestModel):
    # Endpoint implementation
    return {"result": "success"} 