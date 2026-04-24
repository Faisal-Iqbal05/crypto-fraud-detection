# CryptoShield AI

Production-style cryptocurrency fraud detection web application with a FastAPI backend, React dashboard, real-time simulation, analytics, authentication, explainable AI, and a pre-trained SVM fraud detection model.

## Stack

- Backend: FastAPI, Pandas, NumPy, scikit-learn, bcrypt, JWT
- Frontend: React, Tailwind CSS, Framer Motion, Recharts, Lucide React
- ML: Pre-trained SVM with comparison metrics for Logistic Regression, Decision Tree, Random Forest, and extension-ready XGBoost

## Features

- Real-time transaction fraud prediction
- Explainable AI with top contributing factors
- Glassmorphism fintech dashboard UI
- Sidebar navigation for dashboard, prediction, analytics, and model insights
- Interactive fraud analytics charts
- Login and signup with bcrypt password hashing
- Live simulated transaction stream via WebSocket
- Recent transactions table and alert center
- Exportable CSV and PDF reports

## Project Structure

- `backend/` FastAPI app, auth, APIs, simulation, reports
- `frontend/` React + Tailwind dashboard
- `models/` trained artifact and evaluation outputs
- `data/` training and sample data
- `src/` machine learning training pipeline
- `app.py` convenience launcher for backend development

## Backend Setup

```bash
pip install -r requirements.txt
copy .env.example .env
python app.py
```

Backend runs on `http://localhost:8000`.

## Frontend Setup

```bash
cd frontend
copy .env.example .env
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`.

## Training the Model

Regenerate data and retrain:

```bash
python -m src.train --generate_dataset
```

Artifacts are saved to:

- `models/best_model_artifacts.joblib`
- `models/model_comparison.csv`
- `models/classification_reports.csv`
- `models/plots/`

## API Endpoints

- `POST /api/v1/auth/signup`
- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`
- `POST /api/v1/predict`
- `GET /api/v1/analytics/summary`
- `GET /api/v1/analytics/recent`
- `GET /api/v1/models/comparison`
- `GET /api/v1/models/insights`
- `GET /api/v1/reports/export.csv`
- `GET /api/v1/reports/export.pdf`
- `WS /api/v1/simulate/ws`

## Deployment

`render.yaml` is included for deploying the FastAPI backend and the React frontend on Render.
