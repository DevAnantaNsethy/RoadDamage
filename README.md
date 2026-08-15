# Road Damage Detection — Streamlit Deploy

## Overview
This project is a modern YOLOv8-based road damage detection demo built for Python 3.14. It includes a Streamlit web app (`app_streamlit.py`) that accepts a single image upload and displays detected road damage with confidence scores and severity.

## Local testing
Install dependencies and run the app locally:
```bash
python -m pip install --user -r requirements.txt
streamlit run app_streamlit.py
```

## Streamlit Cloud Deployment

### Files required
- `app_streamlit.py` — Streamlit demo app
- `requirements.txt` — Python dependencies
- `packages.txt` — Linux system packages (for OpenCV and graphics libraries)
- `Procfile` — launch command for Streamlit
- `.streamlit/config.toml` — Streamlit runtime configuration
- `model/best.pt` — trained YOLOv8 model weights (~24 MB)

### Deploy to Streamlit Cloud
1. Push this repository to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Create a new app and select this repository.
4. Set the main file path to `app_streamlit.py` if prompted.
5. Streamlit Cloud will:
   - Install system packages from `packages.txt`
   - Install Python dependencies from `requirements.txt`
   - Run `streamlit run app_streamlit.py`

### Notes
- The model requires system libraries (libsm6, libxext6, etc.) which are installed via `packages.txt`
- OpenCV is configured in headless mode for Linux environments
- First app start may take 2-3 minutes while PyTorch and Ultralytics are downloaded
- Model file `model/best.pt` (~24 MB) must be included in the repository

## App Features
- **Image Upload**: Upload any road image (JPG, JPEG, PNG)
- **Detection**: Displays annotated image with bounding boxes
- **Confidence Score**: Shows model confidence for each detection (0-1)
- **Severity Estimation**: Auto-classifies damage severity (None/Repair, Low, Medium, High)
- **Download**: Download the annotated result image

## Keep app alive (optional)

Streamlit Cloud may put inactive apps to sleep. To keep the app active, a lightweight scheduled workflow can ping the app URL every 10 minutes.

1. Add the GitHub repository secret `STREAMLIT_APP_URL` with the full app URL (e.g. `https://your-username-xxxxx.streamlitapp.com`).
   - Go to your repository on GitHub -> Settings -> Secrets and variables -> Actions -> New repository secret.
   - Name: `STREAMLIT_APP_URL` Value: your Streamlit app URL

2. The repository includes a GitHub Actions workflow `.github/workflows/ping_app.yml` that will run every 10 minutes and send a GET request to the app URL. Enable the workflow and ensure Actions are allowed for your repo.

3. Note: Frequent wakeups may be subject to Streamlit Cloud usage policies and GitHub Actions minutes limits. If you prefer a third-party uptime monitor, services like UptimeRobot or cron-job.org can also perform periodic pings.
