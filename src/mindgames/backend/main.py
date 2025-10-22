"""
Entry point for running the FastAPI backend server.

Usage:
    uv run python -m mindgames.backend.main

Or with uvicorn directly:
    uv run uvicorn mindgames.backend.app:app --reload
"""

import uvicorn


def main():
    """Run the FastAPI server."""
    uvicorn.run(
        "mindgames.backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
