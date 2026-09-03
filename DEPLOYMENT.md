# Deployment & MLOps Guide

A FastAPI service wraps the trained Random Forest detector behind a `/predict`
endpoint and a small web UI. This guide covers local runs, Docker, an optional
cloud path, and the MLOps practices around it.

![Demo](figures/demo/demo.gif)

---

## 1. Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** for the UI, or call the API directly:

```bash
# health
curl http://localhost:8000/health

# score a connection (attack example)
curl -X POST "http://localhost:8000/predict?threshold=0.5" \
     -H "Content-Type: application/json" \
     -d '{"dur":9e-06,"proto":"sctp","service":"-","state":"INT","spkts":2,
          "sbytes":104,"rate":111111.1,"sttl":254,"sload":46222220,"smean":52,
          "ct_srv_src":1,"ct_state_ttl":2,"ct_dst_ltm":2,"ct_dst_src_ltm":2,
          "sinpkt":0.009}'
# -> {"attack_probability":0.9803,"prediction":1,"label":"attack","threshold":0.5}
```

Interactive API docs (Swagger) are auto-generated at **/docs**.

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Web UI |
| GET | `/health` | Liveness + whether artifacts loaded |
| GET | `/model-info` | Model type, feature count, threshold, reported metrics |
| POST | `/predict?threshold=` | Score one connection → probability + label |
| GET | `/docs` | Swagger UI |

## 2. Run with Docker

```bash
docker build -t nad-detector .
docker run -p 8000:8000 nad-detector
```

Configuration via environment variables (also honoured by `uvicorn`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_NAME` | `best_model` | `best_model` (Random Forest) or `xgboost` |
| `THRESHOLD` | `0.5` | Decision threshold (lower → higher recall) |
| `MODEL_DIR` | `/app/models` | Where the artifacts live |

```bash
docker run -p 8000:8000 -e MODEL_NAME=xgboost -e THRESHOLD=0.35 nad-detector
```

## 3. Optional — cloud

The container is portable to any managed runtime:

- **AWS** — push to ECR, deploy on ECS Fargate or App Runner; or wrap the model in a **SageMaker** endpoint.
- **GCP** — deploy the image to **Cloud Run**, or register the model in **Vertex AI**.
- **Azure** — **Container Apps** or an **Azure ML** managed endpoint.

All read the same env-var configuration above; no code changes required.

---

## 4. MLOps practices

**Reproducible environments** — pinned `requirements.txt` and a slim
`Dockerfile` give identical runs locally, in CI, and in the cloud.

**Config-driven runs** — `config.yaml` centralises data paths, the selection
metric order, the serving model/threshold, and the drift-watch list.

**Experiment tracking** — metrics for every run are written to
`reports/step4_results.json`; drop-in MLflow/W&B logging can wrap `src/train.py`
to version runs, params, and artifacts.

**CI checks** — `.github/workflows/ci.yml` runs `ruff` lint and the `pytest`
suite (`tests/test_api.py`) on every push and pull request.

**Monitoring plan** — track, per batch of scored traffic:
- input **drift** via PSI on the watch-list features
  (`ct_src_ltm`, `ct_src_dport_ltm`, `ct_dst_src_ltm`, `rate` …); alert when PSI > 0.25;
- prediction-rate and score distribution (a sudden shift can signal drift or evasion);
- **explanation drift** — unexpected movement in SHAP feature ranking as an early adversarial signal.

**Versioning & rollback** — models are saved as versioned artifacts in
`models/`. To roll back, redeploy the previous `best_model.joblib` (or pin an
image tag). Because inference is a pure function of *(preprocessor, model)*,
swapping artifacts is a safe, reversible operation. Keep the fitted
`preprocessor.joblib` versioned **with** its model — they must always match.

**Retraining trigger** — when drift crosses the threshold
(`config.yaml: monitoring.psi_alert_threshold`), retrain on fresh labelled
traffic, revalidate against the held-out metrics, then redeploy.

---

## 5. Demo

The clip above shows the UI scoring an attack connection (98% → **ATTACK**) and
a normal connection (0% → **NORMAL**). Individual frames are in
[`figures/demo/`](figures/demo/). To record your own screencast, run the app and
capture the browser at http://localhost:8000.
