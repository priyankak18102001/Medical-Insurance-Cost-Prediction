import mlflow
import json,os,joblib
from mlflow.tracking import MlflowClient
print("MLflow installed successfully!")

DATA_PATH = "medical_insurance.csv"   # update path if needed
MODEL_OUTPUT_DIR = "models"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

MLFLOW_EXPERIMENT_NAME = "Insurance_Cost_Prediction"
MLFLOW_REGISTERED_NAME = "InsuranceCostModel"   # model name in model registry (change if desired)
# ----------------------------

import mlflow.sklearn
mlflow.set_experiment("Medical_Insurance_Cost_Prediction")
client = MlflowClient() 

best_rmse = float("inf")
best_run = None
best_model_name = None
best_model_uri = None

# 5) Loop over models, log runs
for name, estimator in models.items():
    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id

        # Train
        estimator.fit(x_train, y_train)
        preds = estimator.predict(x_test)

        # Metrics (y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2  = float(r2_score(y_test, preds))

        # Log parameters (a few hyperparams if available)
        try:
            params = estimator.get_params()
            # log a small subset to keep UI tidy
            for k in list(params.keys())[:10]:
                mlflow.log_param(k, str(params[k]))
        except Exception:
            pass

        # Log metrics
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)

        # Log the model artifact (full sklearn estimator)
        mlflow.sklearn.log_model(estimator, artifact_path="model")

        print(f"[run_id={run_id}] {name} → RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")

        # Keep best by RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_run = run_id
            best_model_name = name
            # The logged artifact URI for this run's model:
            best_model_uri = f"runs:/{run_id}/model"

# 6) Register the best model in Model Registry (if desired)
if best_model_uri is None:
    raise RuntimeError("No model was logged. Check training loop.")

print(f"\nBest run: id={best_run}, model={best_model_name}, rmse={best_rmse:.2f}")
# Create registered model if not exists (safe)
try:
    client.create_registered_model(MLFLOW_REGISTERED_NAME)
    print("Created registered model:", MLFLOW_REGISTERED_NAME)
except Exception:
    # likely already exists
    pass

# Create a new model version from the best run
mv = client.create_model_version(name=MLFLOW_REGISTERED_NAME, source=best_model_uri, run_id=best_run)
print("Registered model version:", mv.version)

# Optionally transition to Production and archive others
try:
    client.transition_model_version_stage(name=MLFLOW_REGISTERED_NAME, version=mv.version, stage="Production", archive_existing_versions=True)
    print(f"Transitioned model {MLFLOW_REGISTERED_NAME} v{mv.version} -> Production")
except Exception as e:
    print("Warning: could not transition model stage:", e)

# 7) Save best model locally for easy loading in Streamlit
# Load model back using mlflow API and save as joblib
best_model_obj = mlflow.sklearn.load_model(best_model_uri)
local_model_path = os.path.join(MODEL_OUTPUT_DIR, f"best_pipeline_{best_model_name}.joblib")
joblib.dump(best_model_obj, local_model_path)
print("Saved best model locally to:", local_model_path)

# 8) Save metrics.json to be used by Streamlit app (optional)
metrics_out = {
    "model_name": best_model_name,
    "best_run_id": best_run,
    "test_rmse": best_rmse,
    "test_mae": mae,
    "test_r2": r2
}
with open(os.path.join(MODEL_OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_out, f, indent=2)
print("Saved metrics.json ->", os.path.join(MODEL_OUTPUT_DIR, "metrics.json"))

print("\nMLflow logging & registry steps completed.")