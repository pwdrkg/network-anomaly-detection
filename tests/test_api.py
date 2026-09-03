"""API tests — run with:  pytest -q"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# genuine samples from the UNSW-NB15 test split
NORMAL = {"dur": 0.0, "proto": "arp", "service": "-", "state": "INT", "spkts": 1,
          "sbytes": 46, "sttl": 0, "smean": 46, "ct_srv_src": 2, "ct_state_ttl": 2,
          "ct_dst_ltm": 2, "ct_src_ltm": 2, "ct_srv_dst": 2, "is_sm_ips_ports": 1,
          "sinpkt": 60000.688}
ATTACK = {"dur": 9e-06, "proto": "sctp", "service": "-", "state": "INT", "spkts": 2,
          "sbytes": 104, "rate": 111111.1072, "sttl": 254, "sload": 46222220.0,
          "smean": 52, "ct_srv_src": 1, "ct_state_ttl": 2, "ct_dst_ltm": 2,
          "ct_dst_src_ltm": 2, "sinpkt": 0.009}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["n_features"] == 69


def test_predict_attack():
    r = client.post("/predict", json=ATTACK)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == 1
    assert body["label"] == "attack"
    assert 0.0 <= body["attack_probability"] <= 1.0


def test_predict_normal():
    r = client.post("/predict", json=NORMAL)
    assert r.status_code == 200
    assert r.json()["label"] == "normal"


def test_threshold_override():
    # a very high threshold should flip a borderline case toward "normal"
    r = client.post("/predict?threshold=0.99", json=NORMAL)
    assert r.status_code == 200
    assert r.json()["threshold"] == 0.99
