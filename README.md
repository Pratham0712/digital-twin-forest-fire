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
Fire Weather Index system (FFMC/DMC/DC/BUI/FWI), adds each zone's recent
satellite fire history and long-run fire climatology, scores wildfire risk per
grid zone using three ML models (Random Forest, XGBoost, CNN+LSTM) trained on
3 real fire seasons of satellite-confirmed detections, projects 2-hour fire
spread with a Cellular Automata simulator seeded from high-risk zones, and
visualises all of it on a live, authenticated, multi-region Streamlit dashboard.

### Model results (held-out 2025 season, never seen in training)

| Model | AUC-ROC | Avg precision | **Alert:** accuracy / fires caught | **Early warning:** accuracy / fires caught |
|---|---|---|---|---|
| Random Forest | 0.863 | 0.243 | 92.3% / 41% | 78.3% / 77% |
| XGBoost | 0.870 | 0.259 | 91.2% / 48% | 74.6% / 84% |
| CNN+LSTM | 0.862 | 0.263 | 91.4% / 49% | 74.6% / 82% |
| XGBoost (v1, weather only) | 0.807 | 0.143 | 92.4% / 22% | 71.9% / 75% |

Both operating points use thresholds chosen on training data only (alert: lowest threshold with ≥92% validation accuracy; early warning: ~80% validation recall).

The v2 features (previous-10-day satellite fire history, 5x5 neighbourhood
activity, leave-one-season-out zone climatology) are computed by one shared
module (`src/ml_models/fire_history.py`) for both training and live
inference, and a test asserts the two paths produce identical values.
Labels exclude FIRMS "static land source" detections (steel plants and kilns
in the Ballari belt were ~10% of all detections); the live feed is cleaned
the same way with a learned mask of those locations.

Raw accuracy is not the target metric: only ~4% of zone-days burn, so
predicting "no fire" everywhere scores 95.6% accuracy while catching nothing, and the weather-only v1 model reaches 92.4% at its alert threshold while catching only 22% of fires.
Ranking quality (AUC-ROC, average precision) and recall at a fixed alert
budget are what matter, and the dashboard's alert tiers are anchored to
measured hit rates (`src/ml_models/risk_index.py`):

| Tier | Zone-days flagged | Fires captured | Hit rate (base rate 4.3%) |
|---|---|---|---|
| EXTREME | 1.3% | 12.6% | 43.2% |
| HIGH or above | 8.7% | 48.4% | 24.2% |
| MODERATE or above | 28.3% | 83.5% | 12.8% |

Beyond the core prediction pipeline, the system also:
- **Persists every refresh** (snapshots, alerts, users, activity) to
  **MySQL** via SQLAlchemy when `DATABASE_URL` is set, with a zero-setup
  SQLite fallback for local development (`src/storage/database.py`).
- **Shows a live Activity Log**: every refresh (with per-stage latency),
  new EXTREME detection, alert email, simulation, scenario, report download,
  sign-in and admin change, newest first, updating every 5 seconds.
- **Emails new EXTREME-risk detections** the moment they first appear, not
  on every repeat refresh (`src/notifications/email_notifier.py`).
- **Refreshes on a schedule independent of the dashboard being open**, via a
  standalone background process (`src/scheduler/scheduler.py`) - the same
  role a cron job or Windows Task Scheduler entry would play.
- **Covers India's fire-prone states**: Karnataka/Western Ghats (0.1° grid,
  validated) plus Uttarakhand, Himachal Pradesh, Odisha, Chhattisgarh, Madhya
  Pradesh, Maharashtra, Andhra Pradesh/Telangana and Assam (0.5° grid), served
  by a pooled pan-India model once `scripts/build_india_regions_dataset.py
  --train` has been run, with metrics reported per state.
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
Persistence   -> src/storage/database.py        (MySQL/SQLite: snapshots, alerts, users, activity_log)
Notifications -> src/notifications/email_notifier.py
Scheduling    -> src/scheduler/scheduler.py      (background refresh loop)
Auth          -> src/auth/auth_gate.py           (login + role gating for the dashboard)
Regions       -> src/regions.py                  (curated Indian fire-prone states)
Model routing -> src/ml_models/model_registry.py (which model serves which region)
```

## Setup

Requires Python 3.11 (TensorFlow does not yet support 3.12+).

```bash
git clone https://github.com/Pratham0712/digital-twin-forest-fire.git
cd digital-twin-forest-fire
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt          # dashboard + scheduler (what deployment uses)
pip install -r requirements-train.txt    # only if you will retrain models / run the tests
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
GOOGLE_MAPS_API_KEY=your_key_here   # Spread Simulation satellite map (Maps JavaScript API)
GOOGLE_MAPS_MAP_ID=                 # optional vector Map ID for 3D tilt / rotation
```

### What-If -> Spread Simulation

Google Cloud APIs for the key in `GOOGLE_MAPS_API_KEY`: **Maps JavaScript API** (map) and, for the
location search, **Geocoding API** or **Places API (New)** (either one; Geocoding is tried first).

1. **What-If Simulator**
   * Weather scenario: temperature, humidity, wind speed, wind direction (the bearing the wind blows
     FROM), seeded hotspots, or a preset.
   * Simulation set-up: a preset forest (Bandipur, Nagarhole, Kudremukh, highest-risk zone) or any place
     found with the Google search box on the map; width x height (100-3000 m) or an area (0.25-5 km²);
     cell size (5 / 10 / **25** / 50 m); simulated duration (1 min - 4 h); ignition (upwind edge, centre,
     downwind edge, or points clicked on the map); grid / boundary visibility.
   * The map shows the focus area as a box you can drag and resize (values snap to whole cells and
     update when you release), the 25 m grid, the simulation domain (dashed) and the ignition.
   * **Apply Scenario & Open Spread Simulation** transfers all of it.
2. **Spread Simulation**
   * **Run Simulation** runs the project's `FireSpreadSimulator` on the *simulation domain*: the focus
     area plus a margin of (CA steps + 1) cells on every side. The CA moves fire at most one cell per
     step, so the fire is limited only by the simulated duration (or fuel), never by the drawn box.
   * Conditions: FFMC / BUI / NDVI of the regional grid zone, the scenario wind, a real DEM sampled on a
     ~90 m lattice (Open-Topo-Data / Open-Meteo, cached in `models/focus_terrain/dem_lattice.csv`).
   * The map plays the result with fire-front flames, merged wind-driven smoke plumes, embers, ash and
     a burn scar. Play / pause / reset / 1x-2x-5x, the simulation-time slider, layer toggles and
     camera presets run in the browser (no Streamlit reruns).
   * Analytics, CSV export of every step and GeoJSON export of the simulated burned cells.
   * The original regional 2-hour projection is kept in an expander.

### Interface

* **Global ticker** (`src/dashboard/ui/global_ticker.py`) on every signed-in page: region, risk level,
  alert zones, regional mean wind, last refresh, latest spread run (labelled SIMULATED) and whether the
  FIRMS / OpenWeatherMap feeds are LIVE or DEMO. With no state loaded it says "DEMO / OFFLINE MODE".
* **Command Center**: hero artwork (`src/dashboard/assets/dashboard_hero.jpg`, an illustrative concept
  image, captioned as such; its painted numbers are not data), live status strip, wildfire alert panel
  (latest spread run, or the model's highest-risk zone, always labelled), key metrics, Explore modules,
  data / system status.
* **Sidebar**: the active page's button uses the blue → purple → pink gradient (from the first paint);
  no other element uses it.
* **Login**: glass card over a blurred copy of the hero artwork. "Create account" explains that accounts
  are issued by an admin on the Admin page; there is no self-registration.
* Streamlit's dark theme is set in `.streamlit/config.toml` so built-in widgets match.

### MySQL (optional - SQLite is used when unset)

Any MySQL 8 host works (Railway, Aiven, TiDB Cloud free tiers, or local):

```
DATABASE_URL=mysql+pymysql://USER:PASSWORD@HOST:PORT/DBNAME
DATABASE_SSL=true                 # if the host requires TLS
# DATABASE_SSL_CA=path/to/ca.pem  # verified TLS, if the host gives a CA file
```

Tables are created automatically on first run. To copy existing local data
across once: `python scripts/migrate_sqlite_to_mysql.py`.

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

# Rebuild the Karnataka training set from the raw FIRMS/Meteostat files,
# then train the 3 ML models + the weather-only baseline
python src/ml_models/build_real_dataset.py
python src/ml_models/train_real.py

# Download + build the other Indian states and train the pan-India model
# (needs FIRMS_MAP_KEY; ~1-2 h first time, cached and resumable after)
python scripts/build_india_regions_dataset.py --train

# Interactive dashboard (login required - default admin/changeme123)
streamlit run src/dashboard/app.py

# Background refresh scheduler - keeps the database/alerts current and
# emails new detections even when nobody has the dashboard open
python -m src.scheduler.scheduler                  # runs forever, ~cron-like
python -m src.scheduler.scheduler --once           # single cycle (use this
                                                    # from cron / Windows
                                                    # Task Scheduler instead)

# Run the test suite (pipeline, features/leakage, storage/auth/notifications)
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
    feature_engineering.py             Canadian FWI System (FFMC/DMC/DC/BUI/FWI)
src/ml_models/
    model_trainer.py                   Random Forest, XGBoost, CNN+LSTM model classes
    fire_history.py                    v2 features: fire history, climatology, static-source mask
    risk_index.py                      Probability -> validated alert tiers
    model_registry.py                  Region -> model routing
    build_real_dataset.py              Real FIRMS + Meteostat -> labelled zone-days (any region)
    train_real.py                      Temporal-holdout training + comparison table
    timeseries_builder.py              Synthetic trailing-history builder for CNN-LSTM
    train.py                           Training pipeline + model comparison table
src/simulation/
    cellular_automata.py               8-neighbour CA wildfire spread simulator
src/digital_twin/
    twin_state.py                      DigitalTwin state manager + AlertEngine +
                                        persist_snapshot_and_notify() hook
src/storage/
    database.py                        SQLAlchemy (MySQL/SQLite): snapshots, alerts_log, users, activity_log
src/notifications/
    email_notifier.py                  SMTP email alert on new EXTREME zones
src/scheduler/
    scheduler.py                       Standalone background refresh loop
src/auth/
    auth_gate.py                       Login form + admin/viewer role gating
src/regions.py                         Curated Indian region presets
src/dashboard/
    app.py                             Streamlit dashboard (Command Center)
    pages/1-5_*.py                     What-If, Spread Sim, History, Briefing, Insights
    pages/6_Admin.py                   Admin-only: users, history, notification setup
    pages/7_Activity_Log.py            Live feed of every system/user action
scripts/
    build_india_regions_dataset.py     Multi-state data download, build and training
    migrate_sqlite_to_mysql.py         One-time copy of local SQLite data into MySQL
tests/
    test_pipeline.py                   Integration + regression tests
    test_fire_history.py               Train/serve parity, leakage rules, India builder
    test_storage_auth_notifications.py Storage/auth/notification/activity-log tests
main.py                                Single entry point, runs the full pipeline
```

## Known limitations

- The Karnataka model is validated on Karnataka data; the other states are
  validated only after `build_india_regions_dataset.py --train` has run
  (the sidebar shows each state's held-out AUC once it has). Custom lat/lon
  boxes are never validated.
- Real NDVI/fuel-moisture rasters are not ingested yet; the NDVI column shown
  is a synthetic placeholder and is deliberately not a model input.
- FWI moisture codes are computed per day from default previous-day values
  rather than a full recursive carry-forward (station data has gaps).
- CA spread uses real/synthetic elevation for slope but does not model
  fire suppression, roads, or firebreaks.
- The 25 m local spread (Spread Simulation map) uses the same CA with a
  local-scale base spread probability (`SystemConfig.local_ca_base_spread_prob`
  = 0.8; the regional CA keeps 0.35) and uniform fuel inside the 500 m area.
  It is not yet validated against observed fire perimeters. Vegetation drawn
  on the map is a procedural representation, not individual real trees.
- Without `DATABASE_URL`, the scheduler and dashboard share one local
  SQLite file; set `DATABASE_URL` for any multi-instance or cloud deployment.

## Deployment

Deployed on Streamlit Community Cloud: [add your live URL here once deployed]

## License

Academic project - BMS College of Engineering, 2026.
