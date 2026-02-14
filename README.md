# LabPulse AI

LabPulse AI is a Streamlit dashboard for predictive lab-reagent planning based on
wastewater surveillance and complementary surveillance signals.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## What the app does

- Loads RKI AMELAG wastewater data
- Generates pathogen-specific synthetic lab demand curves with lag and seasonality
- Builds forecast and stockout risk recommendations
- Merges user-uploaded CSV lab series
- Fuses wastewater / GrippeWeb / ARE / Google Trends into a confidence score
- Renders interactive charts and optional PDF reports
- Sends optional webhook/email alerts

## Deployment

Use Docker:

```bash
docker compose up -d --build
```

## Notes on forecasting behavior

- ML mode uses Prophet when installed and enough training rows are available.
- If ML or external feeds fail, the app falls back to rule/synthetic paths.

## Testing

Minimal unit tests exist in `tests/`:

```bash
python -m unittest discover -s tests
```
