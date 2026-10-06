"""
email_notifier.py - sends an email alert when the Digital Twin detects a
zone entering EXTREME severity that was NOT already extreme on the previous
refresh (cross-question 9: "as soon as an active fire is detected... people
should get a notification via email or message").

Deliberately built on stdlib only (smtplib / email.mime) so it needs no new
dependency and works with any SMTP provider (Gmail with an App Password,
Outlook, a college mail server, etc.) - just environment variables, no code
changes required to switch provider.

This module NEVER raises out of its public send function: a notification
failure (bad credentials, no network, SMTP server down) must never crash a
refresh cycle. It logs the failure and returns False instead.
"""
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Iterable, List

logger = logging.getLogger(__name__)


class EmailNotifier:
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        recipients_raw = os.getenv("ALERT_RECIPIENT_EMAILS", "")
        self.recipients: List[str] = [
            r.strip() for r in recipients_raw.split(",") if r.strip()
        ]

    @property
    def is_configured(self) -> bool:
        return bool(
            self.smtp_host and self.smtp_user and self.smtp_password and self.recipients
        )

    def send_new_extreme_alert(self, region_name: str, new_alerts: Iterable) -> bool:
        """new_alerts: iterable of ZoneAlert-like objects (zone_id, latitude,
        longitude, risk_score, reason attributes) that are newly EXTREME
        this refresh cycle. Returns True only if the email was actually
        sent; False on any failure or if not configured (never raises)."""
        new_alerts = list(new_alerts)
        if not new_alerts:
            return False
        if not self.is_configured:
            logger.info(
                "Email notifications not configured (SMTP_HOST/USER/PASSWORD/"
                "ALERT_RECIPIENT_EMAILS) - skipping %d new EXTREME alert(s) for %s.",
                len(new_alerts), region_name,
            )
            return False

        subject = f"[Forest Fire Digital Twin] {len(new_alerts)} new EXTREME risk zone(s) - {region_name}"
        lines = [
            f"The Digital Twin has flagged {len(new_alerts)} zone(s) as newly EXTREME "
            f"risk in {region_name}. This means these zones were NOT extreme on the "
            f"previous refresh - this is a new detection, not a repeat.\n",
        ]
        for a in new_alerts:
            lines.append(
                f"  - Zone {a.zone_id}  (lat {a.latitude:.3f}, lon {a.longitude:.3f})\n"
                f"      Risk score: {a.risk_score:.2f}\n"
                f"      Reason: {a.reason}\n"
            )
        lines.append(
            "\nThis is an automated message from the Digital Twin Framework for "
            "Forest Fire Prediction. Do not reply to this email."
        )
        body = "\n".join(lines)

        try:
            msg = MIMEMultipart()
            msg["From"] = self.smtp_user
            msg["To"] = ", ".join(self.recipients)
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.smtp_user, self.recipients, msg.as_string())
            logger.info(
                "Sent EXTREME-alert email for %d zone(s) in %s to %d recipient(s).",
                len(new_alerts), region_name, len(self.recipients),
            )
            return True
        except (smtplib.SMTPException, OSError) as exc:
            logger.warning("Email notification failed (%s) - continuing without it.", exc)
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    class _FakeAlert:
        def __init__(self, zone_id, lat, lon, score, reason):
            self.zone_id, self.latitude, self.longitude = zone_id, lat, lon
            self.risk_score, self.reason = score, reason

    notifier = EmailNotifier()
    print(f"is_configured: {notifier.is_configured}")
    fake = [_FakeAlert("Z-0042", 13.35, 75.2, 0.91, "extreme FWI (78.4); high wind (11.2 m/s)")]
    sent = notifier.send_new_extreme_alert("Karnataka Western Ghats", fake)
    print(f"send result: {sent} (False is expected with no SMTP_* env vars set)")
