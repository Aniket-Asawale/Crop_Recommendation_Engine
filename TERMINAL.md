# Crop Recommendation Engine — Terminal Commands

## Prerequisites

- Python 3.11+
- All dependencies installed (`pip install -r requirements.txt`)

---

## 1. Start the API Server (FastAPI + Uvicorn)

```bash
cd Crop_Recommendation_Engine
python -c "import uvicorn; uvicorn.run('api:app', host='127.0.0.1', port=8001)"
```

The API will be available at **http://127.0.0.1:8001**

- Health check: `http://127.0.0.1:8001/health`
- Docs (Swagger): `http://127.0.0.1:8001/docs`

## 2. Start the Streamlit Dashboard

> **Note:** The API server (step 1) must be running first.

```bash
cd Crop_Recommendation_Engine
streamlit run app.py --server.port 8501
```

The dashboard will be available at **http://127.0.0.1:8501**

---

## Data Generation & Model Training

Run these commands **in order** if you need to regenerate data or retrain the model.

### 3. Generate Dataset

```bash
cd Crop_Recommendation_Engine
python generators/regenerate_all.py
```

This generates ~7200 rows across all agro-zones and merges them into
`data/datasets/crop_recommendation_dataset.csv`.

### 4. Run Preprocessing Pipeline

```bash
cd Crop_Recommendation_Engine
python preprocessing.py
```

Outputs processed features to `data/processed/features.csv`.

### 5. Train the Model

```bash
cd Crop_Recommendation_Engine
python models/baseline_models.py
```

Trains RF, XGBoost, LightGBM, SVM, KNN, and a Voting ensemble.
Best model is saved to `models/model_registry/best_model_<stamp>.pkl`.

### 6. Run Inference Demo (optional)

```bash
cd Crop_Recommendation_Engine
python models/inference.py
```

Runs 3 test scenarios and writes results to `inference_demo_results.txt`.

---

## Quick Start (All-in-One)

### Option A — Batch Scripts (Recommended)

**Start everything** (API + Dashboard + Cloudflare tunnel):
```bash
cd Crop_Recommendation_Engine
start_crop_cloudflared.bat
```

**Stop everything:**
```bash
cd Crop_Recommendation_Engine
stop_crop.bat
```

### Option B — Manual (two terminals)

**Terminal 1 — API:**
```bash
cd Crop_Recommendation_Engine
python -c "import uvicorn; uvicorn.run('api:app', host='127.0.0.1', port=8001)"
```

**Terminal 2 — Dashboard:**
```bash
cd Crop_Recommendation_Engine
streamlit run app.py --server.port 8501
```

Then open **http://127.0.0.1:8501** in your browser.

---

## Remote Access (Cloudflare)

The `start_crop_cloudflared.bat` script automatically starts a Cloudflare quick tunnel.
The public URL is displayed in the terminal output.

To manually start a tunnel:
```bash
tools\cloudflared.exe tunnel --url http://localhost:8501
```

---

## Ports

| Service    | Port | URL                        |
|------------|------|----------------------------|
| FastAPI    | 8001 | http://127.0.0.1:8001      |
| Streamlit  | 8501 | http://127.0.0.1:8501      |
