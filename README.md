# Heart Disease Classification — MLOps Assignment 01

> Educational ML demo using the UCI Cleveland heart-disease dataset. It is not a medical diagnostic tool.

## Architecture

```mermaid
flowchart LR
  A[UCI Cleveland CSV] --> B[Cleaning and EDA]
  B --> C[Preprocessing pipeline]
  C --> D[Logistic Regression + Random Forest]
  D --> E[Stratified 5-fold CV and held-out metrics]
  E --> F[MLflow experiment tracking]
  E --> G[Selected pipeline: joblib]
  G --> H[FastAPI /predict]
  H --> I[Docker image]
  I --> J[Kubernetes Deployment and Service]
  H --> K[Request logs and /metrics]
  K --> L[Prometheus + Grafana]
  M[GitHub Actions] --> C
  M --> D


## Local setup

Use Python 3.11. From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data and EDA

The training script downloads the processed Cleveland dataset from the UCI URL in `app/train.py`; to use a local copy, pass `--data path/to/processed.cleveland.data`. Missing `?` values are converted to NaN. Features are imputed inside the pipeline to prevent preprocessing leakage. Target values 1–4 are mapped to disease-present class 1; target 0 maps to class 0.

```bash
python app/eda.py
```

Figures are saved under `reports/figures/`. Review class balance and distributions before interpreting model scores.

## Train, tune and track experiments

```bash
python app/train.py
# Quick smoke run for CI
python app/train.py --quick
mlflow ui --backend-store-uri ./mlruns
```

The script compares Logistic Regression and Random Forest using stratified 5-fold cross-validation and GridSearchCV, selects by mean CV ROC-AUC, evaluates the selected configurations on a held-out test split, and logs parameters, metrics, model artifacts, classification reports and plots to local MLflow. The best pipeline is exported to `model/model.joblib`.

## Run the API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Check `GET /health`, `GET /metrics`, and send the JSON in `sample_request.json` to `POST /predict`. Example with curl:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data @sample_request.json
```

The API returns a binary prediction, label, confidence and estimated class-1 probability. The probability is a model score, not a clinical risk assessment.

## Tests and CI

```bash
ruff check app tests
pytest -q
```

GitHub Actions runs linting, unit tests, quick training and uploads reports/MLflow runs as artifacts. Inspect the Actions tab after pushing; capture a successful run and, if relevant, a deliberately failing test run for evidence.

## Docker

```bash
docker build -t heart-api:v1 .
docker run --rm -p 8000:8000 heart-api:v1
```

Capture terminal output for both commands, then capture `/health` and a successful `/predict` request. The container uses the saved pipeline and runs as a non-root user.

## Kubernetes

Ensure `heart-api:v1` is available to your local cluster (for Minikube, build with `minikube image build -t heart-api:v1 .`; for Docker Desktop, build locally). Then:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/heart-api
kubectl get pods -l app=heart-api -o wide
kubectl get svc heart-api
kubectl port-forward svc/heart-api 8080:80
```

In another terminal, test `http://127.0.0.1:8080/health` and `/predict`. A local `LoadBalancer` may need environment-specific support; port-forward is a reproducible fallback. Save manifests and screenshots in `screenshots/`.

## Monitoring

`app/main.py` logs method, path, status and latency and exposes Prometheus metrics at `/metrics`. `k8s/servicemonitor.yaml` is only applicable when the Prometheus Operator CRD is installed and its selector/release label matches your Helm release. For evidence, show `/metrics`, the Prometheus target as UP, and a Grafana panel for request count/latency. If the monitoring stack cannot be installed in time, demonstrate logs and `/metrics` and clearly document this limitation.

## Submission checklist

- [ ] Dataset source/download instructions and EDA plots
- [ ] Two tuned models, 5-fold CV and held-out accuracy/precision/recall/ROC-AUC
- [ ] MLflow runs with parameters, metrics, artifacts and plots
- [ ] Reproducible `requirements.txt` and saved preprocessing/model pipeline
- [ ] Pytest suite and passing GitHub Actions run
- [ ] Docker build/run/predict evidence
- [ ] Kubernetes rollout, pod and endpoint evidence
- [ ] Monitoring metrics/dashboard evidence or documented limitation
- [ ] Ten-page report, architecture diagram and short pipeline video

## Dataset citation

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. *Heart Disease*. University of California, Irvine. https://archive.ics.uci.edu/dataset/45/heart+disease
