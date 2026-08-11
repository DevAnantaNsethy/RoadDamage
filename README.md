# Road Damage Detection — Streamlit Deploy

## Overview
This project is a modern YOLOv8-based road damage detection demo built for Python 3.14. It includes a Streamlit web app (`app_streamlit.py`) that accepts a single image upload and displays detected road damage with confidence scores and severity.

## Deployment
The repository is ready for deployment on Streamlit Cloud.

### Required files
- `app_streamlit.py` — Streamlit demo app
- `requirements.txt` — deployment dependency list
- `Procfile` — launch command for Streamlit
- `.streamlit/config.toml` — Streamlit runtime configuration
- `model/best.pt` — trained YOLOv8 model weights

## Local testing
Install dependencies and run the app locally:
```bash
python -m pip install --user -r requirements.txt
streamlit run app_streamlit.py
```

## Streamlit Cloud
1. Push this repository to GitHub.
2. Create a new Streamlit Cloud app and point it to this repo.
3. Set the main file to `app_streamlit.py` if required.

## Notes
- The demo uses the model weight file under `model/best.pt`.
- Only the Streamlit app is required for deployment; dataset and training scripts are included for maintenance and retraining.
