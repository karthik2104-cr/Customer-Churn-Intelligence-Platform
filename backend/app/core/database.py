"""SQLAlchemy engine and session. SQLite by default, PostgreSQL when configured."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


_engine = None
_session_factory = None


def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_engine():
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings()
        url = settings.resolved_database_url()
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args, future=True)
        if url.startswith("sqlite"):
            event.listen(_engine, "connect", _enable_sqlite_foreign_keys)
        _session_factory = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)
    return _engine


def session_factory():
    get_engine()
    return _session_factory


def get_session() -> Generator[Session, None, None]:
    factory = session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    from app.models import tables  # noqa: F401

    Base.metadata.create_all(bind=get_engine())


def reset_db_state() -> None:
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
