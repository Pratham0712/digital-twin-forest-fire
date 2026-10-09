"""Login screen styling: cinematic background (a blurred copy of the hero
artwork under a dark overlay) and a glass authentication card. Presentation
only; the sign-in itself stays in src/auth/auth_gate.require_login."""
from src.dashboard.ui.assets import LOGIN_BACKGROUND, data_uri


def login_css() -> str:
    bg = data_uri(LOGIN_BACKGROUND)
    bg_layer = f", url('{bg}') center / cover no-repeat fixed" if bg else ""
    return """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stSidebarNav"],
[data-testid="stExpandSidebarButton"] { display: none !important; }
[data-testid="stHeader"] { background: transparent !important; }
.stApp { background: linear-gradient(180deg, rgba(6,8,13,0.80), rgba(6,8,13,0.92))""" + bg_layer + """, #080B12 !important; }
.block-container { padding-top: 7vh !important; max-width: 1200px; }
.st-key-login_card { background: rgba(13,17,26,0.72); border: 1px solid rgba(255,255,255,0.10); border-radius: 20px;
    padding: 30px 30px 22px 30px; backdrop-filter: blur(14px) saturate(120%); -webkit-backdrop-filter: blur(14px) saturate(120%);
    box-shadow: 0 30px 80px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.05); animation: fadeSlideIn .5s ease-out both; }
.lg-brand { text-align: center; margin-bottom: 6px; }
.lg-mark { width: 46px; height: 46px; margin: 0 auto 12px auto; border-radius: 13px; display: grid; place-items: center;
    background: linear-gradient(135deg, rgba(255,122,24,0.22), rgba(255,77,28,0.10)); border: 1px solid rgba(255,122,24,0.40); color: #FFB38A; }
.lg-mark svg { width: 24px; height: 24px; }
.lg-title { font-size: 21px; font-weight: 800; letter-spacing: .08em; color: #F4F6F8; }
.lg-inst { font: 700 11px/1.4 var(--mono); letter-spacing: .2em; color: #FF9A5C; margin-top: 5px; }
.lg-sub { font-size: 13px; color: #8A96A6; margin-top: 4px; }
.st-key-login_card [data-baseweb="tab-list"] { justify-content: center; gap: 4px; }
.st-key-login_card [data-baseweb="tab"] { font-size: 12.5px !important; letter-spacing: .12em; text-transform: uppercase; }
.st-key-login_card [data-testid="stForm"] { border: none !important; padding: 4px 0 0 0 !important; }
.st-key-login_card input { background: rgba(8,11,18,0.75) !important; }
.st-key-login_card [data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #FF7A18, #FF4D1C) !important; color: #1A0A02 !important; border: none !important;
    font-weight: 800 !important; letter-spacing: .14em; min-height: 44px; }
.st-key-login_card [data-testid="stFormSubmitButton"] button p { text-transform: uppercase; font-weight: 800 !important; }
.st-key-login_card [data-testid="stFormSubmitButton"] button:hover { box-shadow: 0 10px 26px rgba(255,90,30,0.35) !important; }
.lg-note { font-size: 12.5px; color: #9AA5B4; line-height: 1.6; background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 12px 14px; margin-top: 6px; }
.lg-foot { text-align: center; font-size: 11.5px; color: #6F7B8B; margin-top: 10px; }
</style>
"""


BRAND_HTML = """
<div class="lg-brand">
  <div class="lg-mark"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"
    stroke-linejoin="round" aria-hidden="true"><path d="M12 3c1 3 4 4.5 4 8.5a4 4 0 0 1-8 0c0-1.6.8-2.7 1.6-3.6.3 1.4 1.2 2.1 1.9 2.1-.6-2.5.5-5 .5-7z"/><path d="M5 20h14"/></svg></div>
  <div class="lg-title">FOREST FIRE DIGITAL TWIN</div>
  <div class="lg-inst">BMS COLLEGE OF ENGINEERING</div>
  <div class="lg-sub">Secure Research Command Center</div>
</div>
"""

CREATE_ACCOUNT_HTML = """
<div class="lg-note"><b style="color:#E8EDF3">Accounts are issued by an administrator.</b><br/>
This deployment has no self-registration: an admin creates viewer or admin accounts on the
<b>Admin</b> page, where passwords are stored only as salted PBKDF2 hashes. Ask your project
administrator for an account, then sign in on the first tab.</div>
"""

FOOT_HTML = '<div class="lg-foot">Session-based sign-in · activity is logged · ISE Batch 42</div>'
