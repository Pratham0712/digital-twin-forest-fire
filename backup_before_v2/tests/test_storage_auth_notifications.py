"""
Tests for the new persistence, auth, and notification layer built to answer
the cross-question review list: SQLite storage (Q6/Q11), role-based
authentication (Q14), and the "new EXTREME zone" notification hook (Q9).

Each test gets its own temp DB (monkeypatched DB_PATH) so this suite never
touches the real data/digital_twin.db and can run in any order or in CI.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest

from src.storage import database as db
from src.notifications.email_notifier import EmailNotifier


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_digital_twin.db")
    db.init_db()
    yield


class _FakeAlert:
    def __init__(self, zone_id, severity, score=0.9):
        self.zone_id, self.severity = zone_id, severity
        self.latitude, self.longitude = 13.0, 75.0
        self.risk_score, self.reason = score, "test reason"


# --------------------------------------------------------------------------- #
# Auth / users
# --------------------------------------------------------------------------- #

def test_create_and_verify_user():
    assert db.create_user("alice", "s3cret123", "viewer") is True
    assert db.verify_user("alice", "s3cret123") == "viewer"
    assert db.verify_user("alice", "wrong_password") is None
    assert db.verify_user("nonexistent_user", "whatever") is None


def test_duplicate_username_rejected():
    assert db.create_user("bob", "pw1", "viewer") is True
    assert db.create_user("bob", "pw2", "admin") is False


def test_role_must_be_admin_or_viewer():
    with pytest.raises(ValueError):
        db.create_user("eve", "pw", "superuser")


def test_ensure_default_admin_only_creates_once():
    db.ensure_default_admin()
    assert len(db.list_users()) == 1
    db.create_user("someone_else", "pw123456", "viewer")
    db.ensure_default_admin()  # should be a no-op now (table not empty)
    assert len(db.list_users()) == 2


def test_set_user_role_and_delete_user():
    db.create_user("carol", "pw123456", "viewer")
    assert db.verify_user("carol", "pw123456") == "viewer"
    assert db.set_user_role("carol", "admin") is True
    assert db.verify_user("carol", "pw123456") == "admin"
    assert db.delete_user("carol") is True
    assert db.verify_user("carol", "pw123456") is None


# --------------------------------------------------------------------------- #
# Snapshots / alerts / notification diff logic
# --------------------------------------------------------------------------- #

def _summary(extreme=0, high=0, total_alerts=None):
    total_alerts = total_alerts if total_alerts is not None else extreme + high
    return {
        "timestamp": "2026-01-01T00:00:00+00:00", "total_zones": 100,
        "total_alerts": total_alerts,
        "severity_breakdown": {"EXTREME": extreme, "HIGH": high},
        "max_risk_score": 0.9, "mean_risk_score": 0.3,
    }


def test_save_snapshot_persists_alerts():
    alerts = [_FakeAlert("Z1", "EXTREME"), _FakeAlert("Z2", "HIGH")]
    snap_id = db.save_snapshot("Region", _summary(1, 1), alerts, offline=True)
    assert snap_id == 1
    recent = db.get_recent_snapshots("Region")
    assert len(recent) == 1
    assert recent[0]["total_alerts"] == 2
    stored_alerts = db.get_recent_alerts("Region")
    assert {a["zone_id"] for a in stored_alerts} == {"Z1", "Z2"}


def test_previously_extreme_empty_with_one_snapshot():
    db.save_snapshot("Region", _summary(1), [_FakeAlert("Z1", "EXTREME")], offline=True)
    assert db.get_previously_extreme_zone_ids("Region") == set()


def test_previously_extreme_reflects_prior_snapshot_only():
    db.save_snapshot("Region", _summary(1), [_FakeAlert("Z1", "EXTREME")], offline=True)
    db.save_snapshot("Region", _summary(2),
                      [_FakeAlert("Z1", "EXTREME"), _FakeAlert("Z2", "EXTREME")], offline=True)
    # looking back from snapshot 2, snapshot 1's extreme set was {Z1}
    assert db.get_previously_extreme_zone_ids("Region") == {"Z1"}


def test_regions_are_independent():
    db.save_snapshot("RegionA", _summary(1), [_FakeAlert("A1", "EXTREME")], offline=True)
    db.save_snapshot("RegionB", _summary(1), [_FakeAlert("B1", "EXTREME")], offline=True)
    assert db.get_recent_snapshots("RegionA")[0]["region_name"] == "RegionA"
    assert {a["zone_id"] for a in db.get_recent_alerts("RegionA")} == {"A1"}
    assert {a["zone_id"] for a in db.get_recent_alerts("RegionB")} == {"B1"}


def test_mark_alerts_notified():
    db.save_snapshot("Region", _summary(1), [_FakeAlert("Z1", "EXTREME")], offline=True)
    alerts = db.get_recent_alerts("Region")
    db.mark_alerts_notified([a["id"] for a in alerts])
    updated = db.get_recent_alerts("Region")
    assert all(a["notified"] == 1 for a in updated)


# --------------------------------------------------------------------------- #
# Email notifier - never raises, correctly reports "not configured"
# --------------------------------------------------------------------------- #

def test_email_notifier_unconfigured_returns_false(monkeypatch):
    for var in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "ALERT_RECIPIENT_EMAILS"):
        monkeypatch.delenv(var, raising=False)
    notifier = EmailNotifier()
    assert notifier.is_configured is False
    result = notifier.send_new_extreme_alert("Region", [_FakeAlert("Z1", "EXTREME")])
    assert result is False


def test_email_notifier_no_op_on_empty_alert_list():
    notifier = EmailNotifier()
    assert notifier.send_new_extreme_alert("Region", []) is False


def test_email_notifier_configured_flag(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "hunter2")
    monkeypatch.setenv("ALERT_RECIPIENT_EMAILS", "a@example.com,b@example.com")
    notifier = EmailNotifier()
    assert notifier.is_configured is True
    assert notifier.recipients == ["a@example.com", "b@example.com"]


def test_email_notifier_send_failure_is_swallowed(monkeypatch):
    """Even when configured, an SMTP connection failure (no real server at
    that host) must be caught and return False, never raise - a notification
    problem must never crash a refresh cycle."""
    monkeypatch.setenv("SMTP_HOST", "smtp.invalid.nonexistent.example")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "hunter2")
    monkeypatch.setenv("ALERT_RECIPIENT_EMAILS", "a@example.com")
    notifier = EmailNotifier()
    result = notifier.send_new_extreme_alert("Region", [_FakeAlert("Z1", "EXTREME")])
    assert result is False
