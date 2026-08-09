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
(OpenWeatherMap), computes the full Canadian Forest Fire Weather Index
system (FFMC/DMC/DC/BUI/FWI) plus an NDVI vegetation-dryness proxy, scores
wildfire risk per grid zone using three ML models (Random Forest, XGBoost,
CNN+LSTM), projects 2-hour fire spread with a Cellular Automata simulator
seeded from high-risk zones, and visualises all of it on a live Streamlit
dashboard.

## Architecture

```
Layer 1: Data Acquisition     -> src/data_ingestion/
Layer 2: Data Processing      -> src/data_processing/
Layer 3: ML Prediction        -> src/ml_models/
Layer 4: CA Spread Simulation -> src/simulation/
Layer 5: Digital Twin + UI    -> src/digital_twin/, src/dashboard/
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

## Running it

```bash
# Full pipeline, one shot, prints a summary
python main.py

# Full pipeline, retraining models first
python main.py --train

# Train the 3 ML models and save a comparison table
python src/ml_models/train.py

# Interactive dashboard
streamlit run src/dashboard/app.py

# Run the test suite
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
    twin_state.py                      DigitalTwin state manager + AlertEngine
src/dashboard/
    app.py                             Streamlit dashboard
tests/test_pipeline.py                 Integration + regression test suite
main.py                                Single entry point, runs the full pipeline
```

## Known limitations

- ML training currently uses a synthetic/rule-derived label in offline mode;
  real historical FIRMS-validated labels are required before results are
  publication-grade (see project paper for the historical validation methodology).
- Cellular Automata model assumes flat terrain; slope-driven spread is not modelled.
- NDVI is a synthetic proxy in offline mode; production deployment should use
  Sentinel-2/MODIS NDVI rasters.

## Deployment

Deployed on Streamlit Community Cloud: [add your live URL here once deployed]

## License

Academic project - BMS College of Engineering, 2026.
