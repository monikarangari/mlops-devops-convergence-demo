# MLOps + DevOps Convergence Demo

End-to-end **AI + DevOps pipeline** using **GitLab CI/CD, Docker, Helm, and FluxCD** — trains an ML model, packages it as a FastAPI service, deploys to **Kubernetes** via **GitOps**, and exposes Prometheus metrics.

---

## 🧩 Tech Stack
- **CI/CD:** GitLab  
- **Model Tracking:** MLflow  
- **Containerization:** Docker  
- **Deployment:** Helm + FluxCD (GitOps)  
- **Orchestration:** Kubernetes  
- **App:** FastAPI (inference + metrics)  
- **Monitoring:** Prometheus + Grafana  

---

## ⚙️ Pipeline Flow
Dev → GitLab CI/CD → Docker Image → GitOps Repo → FluxCD → Kubernetes → Prometheus


---

## 🧠 Key Features
- Automated ML model training & versioning  
- CI/CD pipeline builds and pushes container image  
- GitOps deployment with Helm & FluxCD  
- Canary rollout + instant rollback  
- Live monitoring of predictions & drift  

---

## 🚀 Quick Start
```bash
# Clone repo
git clone https://github.com/<your-username>/mlops-devops-convergence-demo.git
cd mlops-devops-convergence-demo

# Train model & test service
pip install -r requirements.txt
python src/train.py
uvicorn src.infer_service:app --port 8080

# Build container
docker build -t model-svc:latest .

# Deploy to Kubernetes
helm install model-svc ./helm/model-svc -n model-infer --create-namespace

Metrics Exposed

predictions_total

prediction_latency_seconds

drift_psi_feature

model_f1_macro