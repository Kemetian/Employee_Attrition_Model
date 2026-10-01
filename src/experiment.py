import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn as ml
import sys
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Experiment setup and configuration
config = {
    "model_type": "logistic_regression",   # logistic_regression, random_forest, gradient_boosting
    "test_size": 0.4,
    "random_state": 42,
    "handle_missing": "drop",            # median, drop
    "scale_features": True,
    "features_to_drop": ['EmployeeNumber', 'EmployeeCount', 'StandardHours'],                # columns to exclude from training

    # Model-specific hyperparameters
    "lr_C": 1.0,                           # logistic regression regularization
    "rf_n_estimators": 100,                # random forest number of trees
    "rf_max_depth": None,                  # random forest max depth (None = unlimited)
    "gb_n_estimators": 100,                # gradient boosting number of trees
    "gb_learning_rate": 0.1,               # gradient boosting learning rate
    "gb_max_depth": 3,                     # gradient boosting max depth
}

def load_and_prepare_data(config):
    """Load the employee attrition dataset and prepare it for training."""

    url = "https://raw.githubusercontent.com/Kemetian/Employee_Attrition_Model/main/data/IBM-HRAnalytics-Employee-Attrition.csv"
    print(f"Loading data from URL...")
    df = pd.read_csv(url, on_bad_lines='warn')
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Drop any user-specified columns
    if config["features_to_drop"]:
        df = df.drop(columns=config["features_to_drop"], errors="ignore")
        print(f"Dropped features: {config['features_to_drop']}")

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if config["handle_missing"] == "median":
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        print("Filled missing values with median")
    elif config["handle_missing"] == "drop":
        before = len(df)
        df = df.dropna()
        print(f"Dropped rows with missing values: {before} -> {len(df)}")

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=["string"]).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Separate features and target
    features = df.drop(columns=["Attrition"])
    target = df["Attrition"]

    return features, target, len(df), numeric_cols, categorical_cols

def build_model(config):
    """Create a model based on the config."""

    if config["model_type"] == "logistic_regression":
        return LogisticRegression(
            C=config["lr_C"],
            random_state=config["random_state"],
            max_iter=1000
        )
    elif config["model_type"] == "random_forest":
        return RandomForestClassifier(
            n_estimators=config["rf_n_estimators"],
            max_depth=config["rf_max_depth"],
            random_state=config["random_state"]
        )
    elif config["model_type"] == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=config["gb_n_estimators"],
            learning_rate=config["gb_learning_rate"],
            max_depth=config["gb_max_depth"],
            random_state=config["random_state"]
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    print(f"Built model: {config['model_type']}")

def run_experiment(config):
    """Run a single experiment with the given config, tracked by MLflow."""

    # Set the experiment name so all runs are grouped together
    mlflow.set_experiment("employee-attrition-prediction")

    # Start an MLflow run
    with mlflow.start_run():

        # ── Log all configuration as parameters ──
        mlflow.log_param("model_type", config["model_type"])
        mlflow.log_param("test_size", config["test_size"])
        mlflow.log_param("random_state", config["random_state"])
        mlflow.log_param("handle_missing", config["handle_missing"])
        mlflow.log_param("scale_features", config["scale_features"])
        mlflow.log_param("features_dropped", str(config["features_to_drop"]))

        # Log model-specific hyperparameters based on model type
        if config["model_type"] == "logistic_regression":
            mlflow.log_param("C", config["lr_C"])
        elif config["model_type"] == "random_forest":
            mlflow.log_param("n_estimators", config["rf_n_estimators"])
            mlflow.log_param("max_depth", str(config["rf_max_depth"]))
        elif config["model_type"] == "gradient_boosting":
            mlflow.log_param("n_estimators", config["gb_n_estimators"])
            mlflow.log_param("learning_rate", config["gb_learning_rate"])
            mlflow.log_param("max_depth", config["gb_max_depth"])

        # ── Load and prepare data ──
        features, target, n_rows, numeric_cols, categorical_cols = load_and_prepare_data(config)

        mlflow.log_param("n_rows", n_rows)
        mlflow.log_param("n_features", features.shape[1])

        # ── Split the data into training and test sets ──
        features_train, features_test, target_train, target_test = train_test_split(
            features, target,
            test_size=config["test_size"],
            random_state=config["random_state"],
            stratify=target
        )

        # ── Optionally scale features ──
        if config["scale_features"]:
            scaler = StandardScaler()
            features_train = pd.DataFrame(scaler.fit_transform(features_train), columns=features_train.columns)
            features_test = pd.DataFrame(scaler.transform(features_test), columns=features_test.columns)

        # ── Train ──
        model = build_model(config)
        print(f"\nTraining {config['model_type']}...")
        model.fit(features_train, target_train)

        # ── Evaluate ──
        features_pred = model.predict(features_test)
        features_prob = model.predict_proba(features_test)[:, 1]

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

        # ── Log the trained model as an artifact ──
        mlflow.sklearn.log_model(model, "model")

        # ── Log the config file as an artifact for reference ──
        config_path = "config_snapshot.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        mlflow.log_artifact(config_path)
        os.remove(config_path)  # clean up temp file

        # ── Print results ──
        print(f"\n{'='*50}")
        print(f"Model:     {config['model_type']}")
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

if __name__ == "__main__":
    run_experiment(config)
