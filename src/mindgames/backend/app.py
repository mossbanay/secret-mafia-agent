"""
FastAPI application for mindgames arena backend.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from mindgames.db import init_db, get_session, Repository
from .routes import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Lifespan context manager for FastAPI app.

    Initializes database on startup.
    """
    # Startup: Initialize database (uses DATABASE_URL from constants or env var)
    print("Initializing database...")
    init_db()
    print("Database initialized successfully!")

    yield

    # Shutdown: Nothing to do for now
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="MindGames Arena API",
    description="REST API for mindgames arena system - game management, agent tracking, and analytics",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS for NextJS frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # NextJS dev server
        "http://localhost:3001",  # NextJS dev server (alternate port)
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


# Dependency to get database session
def get_db_session() -> Session:
    """
    Dependency that provides a database session.

    Yields:
        SQLAlchemy session instance
    """
    with get_session() as session:
        yield session


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "MindGames Arena API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check(session: Session = Depends(get_db_session)):
    """
    Health check endpoint.

    Verifies database connectivity and returns system status.
    """
    try:
        # Test database connection by executing a simple query
        from sqlalchemy import text

        session.execute(text("SELECT 1"))

        # Try to get counts of key entities
        repo = Repository(session)
        models_count = len(repo.get_active_models())
        kinds_count = len(repo.get_active_kinds())
        prompts_count = len(repo.get_active_prompts())

        return {
            "status": "healthy",
            "database": "connected",
            "active_models": models_count,
            "active_kinds": kinds_count,
            "active_prompts": prompts_count,
        }
    except Exception as e:
        # Return unhealthy but still 200 status so monitoring knows endpoint works
        return {
            "status": "unhealthy",
            "database": "error",
            "error": str(e),
        }
