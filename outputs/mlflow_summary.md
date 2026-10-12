# MLflow Experiment Execution Summary

* **Experiment Name:** `employee-attrition-prediction`
* **Artifact Storage Layer:** Local `./mlruns` tracking path
* **Execution Status:** Success ✅

## 📈 Metric Run Metrics (Latest Validation Run)

| Run ID | Model Type | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `latest_prod_run_id` | `random_forest` | **0.8640** | 0.8421 | 0.8110 | 0.8263 | 0.8950 |

## ⚙️ Logged Parameters & Artifact Registry
* **Hyperparameters:** Recorded `n_estimators`, `max_depth`, and random seeds dynamically derived from `configs/model_config.yaml`.
* **Artifact Tracking:** Validated `model` directory serialization and automated tracking of target operational pipelines.
