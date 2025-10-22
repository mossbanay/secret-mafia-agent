"""
Database session management and initialization.
"""

import os
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .models import Base
from mindgames.constants import DATABASE_URL as DEFAULT_DATABASE_URL


# Global engine instance per process
_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None
_engine_pid: Optional[int] = None  # Track which process created the engine


def get_engine(db_url: Optional[str] = None) -> Engine:
    """
    Get or create the database engine.

    Args:
        db_url: Database URL. If None, reads from DATABASE_URL environment variable.
                Falls back to SQLite if not set.
                Format for PostgreSQL: postgresql://user:password@host:port/dbname

    Returns:
        SQLAlchemy engine instance
    """
    global _engine, _engine_pid

    # Check if we're in a different process than where the engine was created
    current_pid = os.getpid()
    if _engine is not None and _engine_pid != current_pid:
        # We're in a forked process with a stale engine - dispose and recreate
        _engine.dispose()
        _engine = None
        _engine_pid = None

    if _engine is None:
        # Get database URL from parameter, environment, or default from constants
        if db_url is None:
            db_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)

        is_postgres = db_url.startswith("postgresql://") or db_url.startswith(
            "postgres://"
        )
        is_sqlite = db_url.startswith("sqlite://")

        if is_postgres:
            # PostgreSQL configuration optimized for parallel workloads
            _engine = create_engine(
                db_url,
                echo=False,  # Set to True for SQL query logging
                pool_pre_ping=True,  # Verify connections before using them
                poolclass=QueuePool,
                pool_size=10,  # Number of connections to maintain
                max_overflow=20,  # Additional connections when pool is full
                pool_timeout=30,  # Seconds to wait for connection from pool
                pool_recycle=3600,  # Recycle connections after 1 hour
            )
        elif is_sqlite:
            # SQLite configuration with WAL mode for better concurrency
            _engine = create_engine(
                db_url,
                echo=False,
                pool_pre_ping=True,
                connect_args={"timeout": 30, "check_same_thread": False},
            )

            # Enable foreign keys and WAL mode for SQLite
            @event.listens_for(Engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()
        else:
            raise ValueError(f"Unsupported database URL: {db_url}")

        # Store the PID that created this engine
        _engine_pid = current_pid

    return _engine


def init_db(db_url: Optional[str] = None, drop_all: bool = False) -> Engine:
    """
    Initialize the database schema.

    Args:
        db_url: Database URL. If None, reads from DATABASE_URL environment variable.
        drop_all: If True, drop all tables before creating them (use with caution!)

    Returns:
        SQLAlchemy engine instance
    """
    engine = get_engine(db_url)

    if drop_all:
        Base.metadata.drop_all(engine)

    Base.metadata.create_all(engine)

    return engine


def get_session_factory(db_url: Optional[str] = None) -> sessionmaker:
    """
    Get or create the session factory.

    Args:
        db_url: Database URL. If None, reads from DATABASE_URL environment variable.

    Returns:
        SQLAlchemy session factory
    """
    global _SessionLocal, _engine_pid

    # Check if we need to recreate the session factory due to process change
    current_pid = os.getpid()
    if _SessionLocal is not None and _engine_pid != current_pid:
        _SessionLocal = None

    if _SessionLocal is None:
        engine = get_engine(db_url)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    return _SessionLocal


@contextmanager
def get_session(db_url: Optional[str] = None) -> Generator[Session, None, None]:
    """
    Get a database session context manager.

    Usage:
        with get_session() as session:
            # Use session here
            agents = session.query(Agent).all()

    Args:
        db_url: Database URL. If None, reads from DATABASE_URL environment variable.

    Yields:
        SQLAlchemy session instance
    """
    SessionFactory = get_session_factory(db_url)
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine():
    """Reset the global engine instance (useful for testing)."""
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None
