# Credit Card Fraud Anomaly Detection

A deep learning system that detects fraudulent credit card transactions
using a two-stage ensemble: a PyTorch Autoencoder learns what normal
transactions look like, and its reconstruction error is used as an
additional signal for an XGBoost classifier.

Served via a FastAPI REST API with a Streamlit dashboard for monitoring
and inference. Backed by PostgreSQL and deployable with Docker Compose.

Built as a Deep Learning course project.

## How It Works

**Stage 1 — Autoencoder (PyTorch)**
Trained exclusively on normal (non-fraudulent) transactions. It learns
to compress and reconstruct normal transaction patterns. When it
encounters a fraudulent transaction, it struggles to reconstruct it
accurately, producing a high reconstruction error.

**Stage 2 — XGBoost Classifier**
Takes the original 29 transaction features (V1–V28 + Amount) plus the
reconstruction error from Stage 1 as a 30th feature. Trained with
class-imbalance weighting to handle the heavily skewed fraud/normal ratio.

**Inference Pipeline**
Incoming transactions are preprocessed, passed through the Autoencoder
to generate reconstruction error, then classified by XGBoost. Results
and transaction history are stored in PostgreSQL.

## Project Structure
anomaly_detection/

├── src/

│   ├── server.py           # FastAPI REST API

│   ├── dashboard.py        # Streamlit monitoring dashboard

│   ├── train.py            # Training pipeline

│   ├── preprocess.py       # Data preprocessing and scaling

│   ├── db.py               # Async SQLAlchemy database setup

│   ├── schemas.py          # Pydantic request/response schemas

│   ├── ingest_csv.py       # Load CSV data into PostgreSQL

│   ├── fraud.csv           # Sample fraud transactions

│   ├── normal.csv          # Sample normal transactions

│   └── models/

│       ├── autoencoder.py  # PyTorch Autoencoder definition

│       └── xgboost_model.py # XGBoost wrapper

├── models/

│   ├── autoencoder.pth     # Trained Autoencoder weights

│   ├── xgboost.json        # Trained XGBoost model

│   └── scaler.pkl          # Fitted StandardScaler

├── docker-compose.yml      # PostgreSQL service

└── requirements.txt

## Tech Stack

- **Models:** PyTorch (Autoencoder), XGBoost
- **API:** FastAPI + Uvicorn
- **Dashboard:** Streamlit + Plotly
- **Database:** PostgreSQL via SQLAlchemy (async)
- **Deployment:** Docker Compose
- **Other:** scikit-learn, pandas, NumPy

## Getting Started

### Prerequisites

- Python 3.12+
- Docker and Docker Compose

### 1. Start the database

```bash
docker compose up -d
```

This starts a PostgreSQL instance on port 5432.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get the dataset

Download the Credit Card Fraud Detection dataset from
[Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and
place `creditcard.csv` in the `data/` folder.

### 4. Ingest data into the database

```bash
python src/ingest_csv.py
```

### 5. Train the models

```bash
python src/train.py
```

This trains the Autoencoder on normal transactions, then trains XGBoost
with reconstruction error as a feature. Saves all model files to
`models/`.

### 6. Start the API server

```bash
uvicorn src.server:app --reload
```

API available at [http://localhost:8000](http://localhost:8000).
Docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### 7. Launch the dashboard

```bash
streamlit run src/dashboard.py
```

Dashboard available at [http://localhost:8501](http://localhost:8501).

## API Endpoints

The FastAPI server exposes endpoints for:
- Submitting individual transactions for scoring
- Retrieving transaction history and statistics
- Fetching model performance metrics (accuracy, precision, recall, F1,
  ROC-AUC)
- Viewing confusion matrix, ROC curve, and feature importance

Full interactive docs available at `/docs` when the server is running.

## Dataset

This project uses the
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
dataset from Kaggle (ULB Machine Learning Group). It contains 284,807
transactions with 492 frauds (0.17%). Features V1–V28 are PCA-transformed
for confidentiality. `Amount` and `Class` are the only original columns.

The dataset is not included in this repo due to its size. Download it
from Kaggle and place it in `data/creditcard.csv`.
