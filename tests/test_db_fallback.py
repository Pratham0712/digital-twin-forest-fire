"""If the configured MySQL cannot be reached the app must keep working on the
local SQLite file (and say so) instead of crashing at login."""
import pytest

from src.storage import database as db


@pytest.fixture
def unreachable_mysql(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "mysql+pymysql://u:p@no-such-host.invalid:3306/db")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "local.db")
    monkeypatch.setattr(db, "_INITIALISED", set())
    monkeypatch.setattr(db, "_ENGINES", {})
    monkeypatch.setitem(db._FALLBACK, "reason", None)
    yield
    db._FALLBACK["reason"] = None


def test_unreachable_mysql_falls_back_to_sqlite(unreachable_mysql):
    assert db.backend_name() == "MySQL"
    db.init_db()
    assert db.fallback_reason()                         # reason recorded for the banner
    assert db.backend_name().startswith("SQLite")
    assert db.create_user("demo", "pw12345", "viewer")   # the fallback database really works
    assert db.verify_user("demo", "pw12345") == "viewer"


def test_reset_fallback_retries_the_configured_database(unreachable_mysql):
    db.init_db()
    assert db.fallback_reason()
    db.reset_fallback()
    assert db.fallback_reason() is None and db.backend_name() == "MySQL"


def test_sqlite_errors_are_not_swallowed(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "x.db")
    monkeypatch.setattr(db, "_INITIALISED", set())
    monkeypatch.setattr(db, "_ENGINES", {})
    db.init_db()
    assert db.fallback_reason() is None
