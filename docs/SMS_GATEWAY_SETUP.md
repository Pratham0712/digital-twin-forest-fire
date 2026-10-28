# SMS alerts with your own Android phone (textbee.dev)

The dashboard sends SMS through **textbee.dev**: the textbee app on your Android phone
receives each message from textbee's API and sends it from the phone's own SIM. No
SMS company account is needed, but the phone must stay online.

API used (from https://textbee.dev/docs/api-reference and https://textbee.dev/openapi.json):

| Purpose | Call |
|---|---|
| Send | `POST https://api.textbee.dev/api/v1/gateway/send-sms` with header `x-api-key`, body `{"recipients": ["+91…"], "message": "…", "deviceId": "…"}` |
| Check the phone | `GET …/gateway/devices` (fields `_id`, `enabled`, `isDefault`, `lastHeartbeat`, …) |
| Delivery status | `GET …/gateway/devices/{deviceId}/sms-batch/{smsBatchId}`, message status `pending → dispatched → sent → delivered / failed` |

## 1. Set up textbee (once)

1. Create an account at https://app.textbee.dev/dashboard and **verify your e-mail**
   (sending fails with HTTP 400 until you do).
2. On the Android phone (Android 7.0+), download the app from https://textbee.dev/download
   (it is not on the Play Store). Allow your browser to install unknown apps if Android asks.
3. Open the app → **Get Started** → **Scan QR Code** (the QR code is on the dashboard), or
   **Enter manually instead** and paste an API key → optionally name the device → **Register Device**.
4. Tap **Grant All Permissions**: **Send SMS, Receive SMS, Phone state**, and on Android 13+
   **Notifications**. On Android 15+ you may first need *App info → Allow restricted settings*.
   The last screen shows the **Device ID**.
5. Keep the phone reachable:
   - set the app's battery usage to **Unrestricted** and turn battery saver off;
   - follow your brand's extra steps (autostart / background activity on Xiaomi, Samsung, OPPO, vivo, …);
   - keep it **plugged in**, on Wi-Fi or mobile data, and with an **active SIM with SMS balance**;
   - don't force-stop the app or clear it from recent apps.

   textbee sends a heartbeat every 30 minutes. A last heartbeat under 45 minutes old is healthy;
   the dashboard warns you when it is older.
6. In the dashboard's **API Keys** panel, copy an API key (or use the one from step 3).

## 2. Configure the Forest Fire Digital Twin

Add to your local `.env` file (never commit it; `.gitignore` already excludes it):

```
SMS_PROVIDER=textbee
TEXTBEE_API_KEY=<your key>
TEXTBEE_DEVICE_ID=<optional: Device ID from step 4>
TEXTBEE_BASE_URL=https://api.textbee.dev/api/v1
TEXTBEE_TIMEOUT=15
ALERT_PROVIDER_MODE=live
ALERT_SMS_RECIPIENTS=+91XXXXXXXXXX
```

Numbers must be in international format (`+` and country code, e.g. `+919812345678`).
`ALERT_PROVIDER_MODE=mock` records attempts without sending anything.

Restart the dashboard. You can also enter the key for one session on
**Admin → Notifications → SMS gateway** (kept in memory only, never written to disk).

## 3. Test it

On **Admin → Notifications → SMS gateway**:

1. **Test connection** checks the key and the phone (device enabled, last heartbeat). Nothing is sent.
2. Enter your number and press **Send test SMS**. One SMS marked *TEST ALERT (no fire)* is submitted.
   - "Request accepted by gateway" means textbee accepted it. It does **not** yet mean the phone sent it.
   - Press **Refresh SMS status** after a few seconds. It shows *Sent to the network* or
     *Delivery confirmed by provider* only when textbee's own record says so.
   - Check that the SMS actually arrived on the destination phone.

A test SMS never starts the automatic fire-alert workflow and never changes cooldowns.

## 4. Automatic fire alerts

On **Admin → Notifications → Automatic fire alerts** (stored in the database; **off by default**):

- **Minimum model fire probability** (default 0.35): the XGBoost fire probability of the zone.
  Only EXTREME zones from live data are eligible. Demo/offline data and What-If scenarios never notify.
- **Cooldown** (default 360 min) per incident (region and zone). It is persistent, so it survives
  restarts and is shared with the background scheduler. Inside the cooldown a zone is
  notified again only if its probability rose by at least the **escalation step** (default 0.10).
- **Channels**:
  - *SMS, e-mail only if no SMS was accepted* (default);
  - *SMS and e-mail*;
  - *SMS only*;
  - *E-mail only*.

  SMS and e-mail results are recorded separately in the history.

## 5. Manual e-mail / SMS about a simulation ("🚨 Send Emergency Alert")

After any completed simulation, including a hypothetical What-If run, **Send Emergency Alert** opens a
small *Delivery method* menu with two options. Nothing is sent from the menu itself.

- **✉️ Email Alert** - one real e-mail to `ALERT_RECIPIENT_EMAILS` (plus registered recipients whose e-mail is
  verified on Admin → Emergency alerts) through the SMTP settings. "Accepted by SMTP server" is not proof of inbox
  delivery.
- **📱 SMS to Mobile** - one real SMS to `ALERT_SMS_RECIPIENTS` through the textbee gateway (details below).

Each option sends only its own channel. Both use the same rules:

- **Confirmation first.** A short dialog ("Confirm Email Alert" / "Confirm SMS Alert") shows only the masked
  destination and *Confirm & Send* / *Cancel*. The full message is built only when you confirm; *Cancel*, Esc
  or ✕ sends nothing. The result appears as a short toast; nothing is added below the simulation.
- **Labelled as a simulation.** The SMS says `SIMULATION ONLY — HYPOTHETICAL SCENARIO — NOT A CONFIRMED FIRE`;
  the e-mail subject and banner say `SIMULATION TEST ALERT - HYPOTHETICAL SCENARIO - NOT A CONFIRMED FIRE`.
- **Same score as the report.** The report, e-mail (HTML and plain text) and SMS all read this simulation's
  prediction from one shared set of values (`emergency.alert_facts`) and print it with the same function
  (`risk_score.format_score`):
  - *ML Fire Risk Score* `42% (LOW)`: computed by `prediction_service.predict()` -> `RiskScore.score()`
    (`models/risk_score_real.json`), category from `risk_score.category()`. A ranking against historical
    fire-season days, not a probability.
  - *Fire-occurrence probability* `1.41%`: the raw model output, shown separately.
  Missing values are left out, never invented. Automatic alerts show the same stored score for each zone.
- **SMS length.** The emojis make it a Unicode SMS: 70 characters per part, 67 when split. A typical alert is
  about 320 characters, roughly 5 parts. The phone shows one message; your mobile plan may count each part.
- **Separate from automatic alerts.** No confidence threshold applies, and the 6-hour automatic incident
  cooldown is neither checked nor changed.
- **Manual cooldown.** Each simulation and recipient pair has its own cooldown (default **5 min**, set on
  Admin → Notifications, 1-1440 min). The dialog shows the next allowed time; there is no override.
  A definite failure (for example HTTP 401) frees the slot for an immediate retry. An uncertain outcome
  (timeout) keeps it, to avoid a duplicate.
- **Not sent when unready.** It is never sent in mock mode, without a configured gateway, or without a valid
  recipient; the dialog says why.
- **Logged separately.** These messages are logged as `MANUAL SIMULATION` (alert ids `MSIM-…`); delivery
  status and history are on Admin → Notifications and in the Activity Log.
- **Report downloads never send anything.**

## Limits and costs

- **textbee free plan:** 50 SMS per day, 300 per month, 1 active device (https://textbee.dev/docs/pricing).
  Paid plans: Pro US$14.99/month, Scale US$29.99/month. HTTP 429 means a plan limit was reached.
- **SIM charges:** each SMS is an ordinary SMS from your SIM, billed by your mobile operator.
  Many Indian prepaid plans cap SMS per day (often 100), and international SMS may be blocked or expensive.
- **Phone must be online:** if it is asleep, offline or out of signal, messages wait in textbee
  (status *pending*) until it reconnects. For urgent alerts keep e-mail as the fallback.
- **SMS only:** textbee does not send MMS, and iOS is not supported.

## Recipient verification (Admin → Emergency alerts)

Only **e-mail addresses** are verified, with a one-time code sent through the SMTP settings used by the
e-mail alerts: 6 digits, valid 10 minutes, single use, 5 wrong entries lock the code, one new code per minute
and at most 5 per hour; only salted hashes are stored and the code is never shown or logged. A changed address
must be verified again. In mock mode or without SMTP settings nothing is sent and the page says what to fix.

Mobile numbers are **not** verified (SMS verification was removed - Indian carriers often refuse code-like SMS
from an ordinary SIM). A registered recipient receives SMS alerts when the number is a valid mobile number
(10-digit Indian numbers are stored as `+91XXXXXXXXXX`) and "SMS alerts" is ticked; `ALERT_SMS_RECIPIENTS` works
as before. All SMS leave from the one phone paired with textbee.

### "Not sent" on the gateway phone

If an SMS alert appears in the **gateway phone's** Messages app with "Not sent", the app and textbee worked and
the phone, SIM or carrier refused it:

1. Send a normal SMS by hand from the gateway phone to the same number. If that fails too: SIM balance / SMS
   pack, daily SMS cap (many Indian plans allow 100 per day; long or emoji SMS count as several), airplane mode,
   or the carrier.
2. Admin → Notifications → *Send test SMS* to the same number.
3. Android settings for the textbee app: SMS permission allowed, battery "Unrestricted", background data on,
   and on dual-SIM phones the SIM with an SMS plan chosen for SMS.
