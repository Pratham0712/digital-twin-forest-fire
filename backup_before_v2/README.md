# Digital Twin Framework for Forest Fire Prediction

A digital twin system for wildfire risk prediction and spread simulation over
the Karnataka and Western Ghats region, India. Built as a final-year capstone
project at BMS College of Engineering, Department of Information Science
and Engineering.

**Team (Batch 42):** Pratham Manoj Patil (1BM23IS180), Praveen Kumar Y
(1BM23IS181), Raviteja S (1BM23IS193), Rishikesh Bulagouda (1BM23IS198)
**Guide:** Dr. Sreelatha R, Associate Professor, Dept. of ISE

## What it does

Ingests live satellite fire detections (NASA FIRMS) and weather data
(OpenWeatherMap / Meteostat for history), computes the full Canadian Forest
Fire Weather Index system (FFMC/DMC/DC/BUI/FWI) plus an NDVI vegetation-dryness
proxy, scores wildfire risk per grid zone using three ML models (Random
Forest, XGBoost, CNN+LSTM) trained on 3 real fire seasons of satellite-
confirmed detections, projects 2-hour fire spread with a Cellular Automata
simulator seeded from high-risk zones, and visualises all of it on a live,
authenticated, multi-region Streamlit dashboard.

Beyond the core prediction pipeline, the system also:
- **Persists every refresh** (snapshots + alerts) to a real SQLite database,
  not just in-memory state or CSVs (`src/storage/database.py`).
- **Emails new EXTREME-risk detections** the moment they first appear, not
  on every repeat refresh (`src/notifications/email_notifier.py`).
- **Refreshes on a schedule independent of the dashboard being open**, via a
  standalone background process (`src/scheduler/scheduler.py`) - the same
  role a cron job or Windows Task Scheduler entry would play.
- **Supports 5 regions** out of the box (Karnataka/Western Ghats - the one
  the models are actually trained and validated on - plus California,
  Australia, Mediterranean, and any custom lat/lon box), with a live FWI + ML
  prediction generated for whichever region is selected, including one that
  was never in the training data.
- **Gates access behind login**, with an admin role (user management,
  raw snapshot/alert history, notification setup) and a viewer role (full
  read access to every prediction/simulation page).

## Architecture

```
Layer 1: Data Acquisition     -> src/data_ingestion/
Layer 2: Data Processing      -> src/data_processing/
Layer 3: ML Prediction        -> src/ml_models/
Layer 4: CA Spread Simulation -> src/simulation/
Layer 5: Digital Twin + UI    -> src/digital_twin/, src/dashboard/

Supporting services:
Persistence   -> src/storage/database.py        (SQLite: snapshots, alerts, users)
Notifications -> src/notifications/email_notifier.py
Scheduling    -> src/scheduler/scheduler.py      (background refresh loop)
Auth          -> src/auth/auth_gate.py           (login + role gating for the dashboard)
Regions       -> src/regions.py                  (the 5 supported region presets)
```

## Setup

Requires Python 3.11 (TensorFlow does not yet support 3.12+).

```bash
git clone https://github.com/Pratham0712/digital-twin-forest-fire.git
cd digital-twin-forest-fire
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### API keys (optional)

The system runs fully in offline/demo mode with synthetic data if no keys
are set. For real data:

1. Copy `.env.example` to `.env`
2. NASA FIRMS: free key at https://firms.modaps.eosdis.nasa.gov/api/
3. OpenWeatherMap: free key at https://home.openweathermap.org/api_keys
   (activation can take up to 2 hours after signup)

```
FIRMS_MAP_KEY=your_key_here
OWM_API_KEY=your_key_here
```

### Email alerts + default login (optional)

Add to `.env` (see `.env.example` for the full list):

```
SMTP_HOST=smtp.gmail.com
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_gmail_app_password
ALERT_RECIPIENT_EMAILS=someone@example.com
```

Without these, the system still detects and logs new EXTREME zones in the
database, it just won't email anyone. The dashboard itself is always
login-gated: on first run it creates a default account, **admin /
changeme123** - sign in and change the password from the Admin page (or set
`DEFAULT_ADMIN_USERNAME`/`DEFAULT_ADMIN_PASSWORD` in `.env` before first run).

## Running it

```bash
# Full pipeline, one shot, prints a summary
python main.py

# Full pipeline, retraining models first
python main.py --train

# Train the 3 ML models and save a comparison table
python src/ml_models/train_real.py

# Interactive dashboard (login required - default admin/changeme123)
streamlit run src/dashboard/app.py

# Background refresh scheduler - keeps the database/alerts current and
# emails new detections even when nobody has the dashboard open
python -m src.scheduler.scheduler                  # runs forever, ~cron-like
python -m src.scheduler.scheduler --once           # single cycle (use this
                                                    # from cron / Windows
                                                    # Task Scheduler instead)

# Run the test suite (28 tests: pipeline + storage/auth/notifications)
pytest tests/ -v
```

## Project structure

```
config/config.py                       Central configuration (region, thresholds, API endpoints)
src/data_ingestion/
    firms_client.py                    NASA FIRMS satellite hotspot API client
    weather_client.py                  OpenWeatherMap client
    ingestion_module.py                Grid builder + spatial nearest-neighbor join
src/data_processing/
    feature_engineering.py             Canadian FWI System (FFMC/DMC/DC/BUI/FWI) + NDVI
src/ml_models/
    model_trainer.py                   Random Forest, XGBoost, CNN+LSTM model classes
    timeseries_builder.py              Synthetic trailing-history builder for CNN-LSTM
    train.py                           Training pipeline + model comparison table
src/simulation/
    cellular_automata.py               8-neighbour CA wildfire spread simulator
src/digital_twin/
    twin_state.py                      DigitalTwin state manager + AlertEngine +
                                        persist_snapshot_and_notify() hook
src/storage/
    database.py                        SQLite: snapshots, alerts_log, users
src/notifications/
    email_notifier.py                  SMTP email alert on new EXTREME zones
src/scheduler/
    scheduler.py                       Standalone background refresh loop
src/auth/
    auth_gate.py                       Login form + admin/viewer role gating
src/regions.py                         The 5 supported region presets
src/dashboard/
    app.py                             Streamlit dashboard (Command Center)
    pages/1-5_*.py                     What-If, Spread Sim, History, Briefing, Insights
    pages/6_Admin.py                   Admin-only: users, history, notification setup
tests/
    test_pipeline.py                   Integration + regression test suite (14 tests)
    test_storage_auth_notifications.py Storage/auth/notification unit tests (14 tests)
main.py                                Single entry point, runs the full pipeline
```

## Known limitations

- The trained models (`train_real.py`) are validated only against Karnataka/
  Western Ghats data. Other region presets and custom lat/lon boxes still get
  a full, live FWI + ML prediction, but treat that as an architecture/
  transfer demo, not a validated regional forecast, until a model is trained
  on that region's own historical data.
- NDVI is a synthetic-but-label-independent proxy (see `synthetic_ndvi_independent`
  in `feature_engineering.py`); production deployment should use real
  Sentinel-2/MODIS NDVI rasters.
- CA spread uses real/synthetic elevation for slope but does not model
  fire suppression, roads, or firebreaks.
- The background scheduler and dashboard both write to the same local
  SQLite file; a genuinely multi-instance deployment would move this to a
  server-based database (Postgres) - the SQL used here has no SQLite-only
  syntax, so that's a small change, not a rewrite.

## Deployment

Deployed on Streamlit Community Cloud: [add your live URL here once deployed]

## License

Academic project - BMS College of Engineering, 2026.
