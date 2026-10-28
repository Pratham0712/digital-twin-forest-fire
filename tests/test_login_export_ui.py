"""Sign-in layout (Streamlit's in-field Enter hint hidden, our hint below the password, responsive card) and the
Export Report menu (PDF / Word only, no preview or print view, built on click, downloads without a rerun)."""
import inspect

import pytest


def test_login_css_hides_the_in_field_hint_and_keeps_the_form_responsive():
    from src.dashboard.ui.login import login_css
    css = login_css()
    assert '.st-key-login_card [data-testid="InputInstructions"] { display: none !important; }' in css
    assert ".lg-hint" in css and "@media (max-width: 640px)" in css and "max-width: 1100px" in css
    assert '[data-testid="stFormSubmitButton"] { margin-top: 10px; }' in css


def test_login_form_fields_and_submit_unchanged():
    from src.auth import auth_gate
    src = inspect.getsource(auth_gate._render_sign_in)
    for s in ('st.text_input("Username"', 'st.text_input("Password", type="password"', 'st.form_submit_button("Sign in"',
              "db.verify_user(", "lg-hint"):
        assert s in src
    assert "create_user(" not in src                                   # signing in never creates an account


def test_export_is_a_pdf_word_dropdown_without_preview_or_print():
    from src.dashboard.ui import alerts_ui as ui
    assert ui.XP_OPTIONS == (("pdf", "PDF (.pdf)"), ("docx", "Word (.docx)"))
    src = inspect.getsource(ui._export_menu)
    assert 'st.popover("📄 Export Report"' in src and "st.download_button(" in src and 'on_click="ignore"' in src
    assert "rep.export_document(snapshot, k)" in src                      # deferred: built only when clicked
    for gone in ("_export_panel", "EXPORT_PANEL_CSS", "PRINT_BUTTON_HTML", "_report_dialog", "_export_bytes"):
        assert not hasattr(ui, gone), gone
    whole = inspect.getsource(ui)
    assert "render_html" not in whole and "Preview" not in whole and "print()" not in whole


def test_export_document_built_once_per_snapshot_and_format(monkeypatch):
    from src.reports import simulation_report as rep
    calls = []
    monkeypatch.setattr(rep, "render_pdf", lambda s: calls.append("pdf") or b"%PDF-x")
    monkeypatch.setattr(rep, "render_docx", lambda s: calls.append("docx") or b"PK-x")
    rep._DOC_CACHE = None
    snap = {"report_id": "FFDT-ONCE", "generated_utc": "2026-10-17T00:00:00+00:00"}
    assert rep.export_document(snap, "pdf") == rep.export_document(snap, "pdf") == b"%PDF-x"
    assert calls == ["pdf"]                                               # Word never built, PDF once
    assert rep.export_document(snap, "docx") == b"PK-x" and calls == ["pdf", "docx"]
    with pytest.raises(ValueError):
        rep.export_document(snap, "html")
