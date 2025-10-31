from src.infer_service import app
from fastapi.testclient import TestClient
client = TestClient(app)

def test_health():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True

def test_predict_basic():
    r = client.post("/predict", json={"rows":[[5.1,3.5,1.4,0.2]]})
    assert r.status_code == 200
    body = r.json()
    assert "pred" in body and len(body["pred"]) == 1
