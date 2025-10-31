import os, json, joblib
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT", "iris-demo")
MODEL_OUT           = os.getenv("MODEL_OUT", "models/model.joblib")

def main():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        f1 = f1_score(yte, pred, average="macro")

        mlflow.log_metric("f1_macro", f1)
        mlflow.sklearn.log_model(clf, artifact_path="model")
        os.makedirs("models", exist_ok=True)
        joblib.dump(clf, MODEL_OUT)

        # Save a small schema/metadata for the inference service
        meta = {
            "classes": iris.target_names.tolist(),
            "feature_names": iris.feature_names,
            "f1_macro": f1
        }
        with open("models/metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        mlflow.log_artifact("models/metadata.json")

        print(f"Run {run.info.run_id} F1={f1:.4f}; model at {MODEL_OUT}")

if __name__ == "__main__":
    main()
