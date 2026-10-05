import mlflow
import yaml
import os
import mlflow.sklearn as ml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score, f1_score
from train import trained_model, config, features_test, target_test

mlflow.set_experiment("employee-attrition-prediction")

# Connect to the experiment
experiment = mlflow.get_experiment_by_name("employee-attrition-prediction")

def make_predictions(config):
    """Import the model built during the experiment and make predictions on the test set."""

        # ── Evaluate ──
        features_pred = trained_model.predict(features_test)
        features_prob = trained_model.predict_proba(features_test)[:, 1]

        accuracy = accuracy_score(target_test, features_pred)
        precision = precision_score(target_test, features_pred)
        recall = recall_score(target_test, features_pred)
        f1 = f1_score(target_test, features_pred)
        auc = roc_auc_score(target_test, features_prob)

        # ── Log metrics ──
        mlflow.log_metric("accuracy", round(accuracy, 4))
        mlflow.log_metric("precision", round(precision, 4))
        mlflow.log_metric("recall", round(recall, 4))
        mlflow.log_metric("f1_score", round(f1, 4))
        mlflow.log_metric("auc_roc", round(auc, 4))

        # ── Log the config file as an artifact for reference ──
        config_path = "config/model_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, indent=2, default_flow_style=False)
        mlflow.log_artifact(config_path)
        os.remove(config_path)  # clean up temp file

        # ── Print results ──
        print(f"\n{'='*50}")
        print(f"Model:     {config['model']['model_type']}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        print(f"{'='*50}")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        print("View this run in the UI: mlflow ui")

    return run_id
