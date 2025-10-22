# FastAPI Backend

REST API backend for the mindgames arena system.

## Overview

This module provides a FastAPI-based REST API that connects to the SQLAlchemy database layer. Currently implements basic health checking, with room for expansion to full game management and analytics endpoints.

## Features

✅ FastAPI application with automatic OpenAPI documentation
✅ Database session management via dependency injection
✅ Health check endpoint for monitoring
✅ Lifespan management for database initialization

## Quick Start

### Running the Server

**Option 1: Using the main script**
```bash
uv run python -m mindgames.backend.main
```

**Option 2: Using uvicorn directly**
```bash
uv run uvicorn mindgames.backend.app:app --reload
```

**Option 3: With custom host/port**
```bash
uv run uvicorn mindgames.backend.app:app --host 0.0.0.0 --port 8080 --reload
```

The server will start on `http://localhost:8000` by default.

### Testing

Run the test script:
```bash
uv run python -m mindgames.backend.test_api
```

## API Endpoints

### Root Endpoint
- **GET /** - API information

**Response:**
```json
{
  "message": "MindGames Arena API",
  "version": "0.1.0",
  "docs": "/docs"
}
```

### Health Check
- **GET /health** - System health and database status

**Response (healthy):**
```json
{
  "status": "healthy",
  "database": "connected",
  "active_models": 5,
  "active_kinds": 3,
  "active_prompts": 10
}
```

**Response (unhealthy):**
```json
{
  "status": "unhealthy",
  "database": "error",
  "error": "error message here"
}
```

### Interactive Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Module Structure

```
mindgames/backend/
├── __init__.py       # Package exports
├── app.py            # FastAPI application and endpoints
├── main.py           # Server entry point
├── test_api.py       # API tests
└── README.md         # This file
```

## Database Integration

The backend uses dependency injection to provide database sessions to endpoints:

```python
from fastapi import Depends
from sqlalchemy.orm import Session
from mindgames.backend.app import get_db_session
from mindgames.db import Repository

@app.get("/example")
async def example_endpoint(session: Session = Depends(get_db_session)):
    repo = Repository(session)
    # Use repo to query database
    models = repo.get_active_models()
    return {"models": models}
```

## Configuration

The database path is configured in `app.py`:
```python
DB_PATH = "mindgames.db"
```

To use a different database, modify this constant or make it configurable via environment variables.

## Development

### Adding New Endpoints

1. Add endpoint functions to `app.py`
2. Use `Depends(get_db_session)` to get database access
3. Use the `Repository` class for database operations
4. Add tests to `test_api.py`

Example:
```python
@app.get("/models")
async def list_models(session: Session = Depends(get_db_session)):
    """List all active models."""
    repo = Repository(session)
    models = repo.get_active_models()
    return {"models": [{"id": m.id, "name": m.name} for m in models]}
```

### Running with Auto-Reload

For development, use the `--reload` flag:
```bash
uv run uvicorn mindgames.backend.app:app --reload
```

This will automatically restart the server when code changes are detected.

## Next Steps

Potential endpoints to add:
- **Models**: CRUD operations for LLM models
- **Kinds**: CRUD operations for agent kinds
- **Prompts**: CRUD operations for prompts
- **Agents**: Create and query agents
- **Games**: Create games, add turns, query results
- **Analytics**: TrueSkill rankings, performance metrics
- **WebSocket**: Real-time game updates

## Testing

The test script (`test_api.py`) uses FastAPI's `TestClient` for synchronous testing:

```python
from fastapi.testclient import TestClient
from mindgames.backend.app import app

client = TestClient(app)
response = client.get("/health")
assert response.status_code == 200
```

## Deployment

For production deployment:

1. **Using uvicorn with workers:**
   ```bash
   uvicorn mindgames.backend.app:app --host 0.0.0.0 --port 8000 --workers 4
   ```

2. **Using gunicorn with uvicorn workers:**
   ```bash
   gunicorn mindgames.backend.app:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

3. **Behind a reverse proxy (nginx):**
   ```nginx
   location / {
       proxy_pass http://127.0.0.1:8000;
       proxy_set_header Host $host;
       proxy_set_header X-Real-IP $remote_addr;
   }
   ```

## Dependencies

- `fastapi>=0.115.0` - Web framework
- `uvicorn>=0.32.0` - ASGI server
- `sqlalchemy>=2.0.43` - Database ORM
