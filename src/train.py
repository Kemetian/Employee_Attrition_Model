from pdb import run

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
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yaml
from configs.model_config import config
from evaluate import make_predictions

def load_config(model_config: str) -> dict:
    """Reads and parses the YAML configuration file."""
    with open(model_config, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_and_prepare_data(config):
    """Load the employee attrition dataset and prepare it for training."""

    url = config["data"]["url"]
    print(f"Loading data from URL...")
    df = pd.read_csv(url, on_bad_lines='warn')
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Drop any user-specified columns
    if config["data"].get("features_to_drop"):
        df = df.drop(columns=config["data"]["features_to_drop"], errors="ignore")
        print(f"Dropped features: {config['data']['features_to_drop']}")

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if config["model"]["handle_missing"] == "median":
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        print("Filled missing values with median")
    elif config["model"]["handle_missing"] == "drop":
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
    features = df.drop(columns=[config["data"]["target_column"]])
    target = df[config["data"]["target_column"]]

    return features, target, len(df), numeric_cols, categorical_cols

def build_model(config):
    """Create a model based on the config."""

    if config["model"]["model_type"] == "logistic_regression":
        model = LogisticRegression(
            C=config["model"]["lr_C"],
            random_state=config["model"]["random_state"],
            max_iter=1000
        )
        return model
    elif config["model"]["model_type"] == "random_forest":
        model = RandomForestClassifier(
            n_estimators=config["model"]["rf_n_estimators"],
            max_depth=config["model"]["rf_max_depth"],
            random_state=config["model"]["random_state"]
        )
        return model
    elif config["model"]["model_type"] == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=config["model"]["gb_n_estimators"],
            learning_rate=config["model"]["gb_learning_rate"],
            max_depth=config["model"]["gb_max_depth"],
            random_state=config["model"]["random_state"]
        )
        return model
    else:
        raise ValueError(f"Unknown model type: {config['model']['model_type']}")

    print(f"Built model: {config['model']['model_type']}")

def train_model(config):
    """Train the built model, tracked by MLflow."""

    # Set the experiment name so all runs are grouped together
    mlflow.set_experiment("employee-attrition-prediction")

    # Start an MLflow run
    with mlflow.start_run():

        # ── Log all configuration as parameters ──
        mlflow.log_param("model_type", config["model"]["model_type"])
        mlflow.log_param("test_size", config["model"]["test_size"])
        mlflow.log_param("random_state", config["model"]["random_state"])
        mlflow.log_param("handle_missing", config["model"]["handle_missing"])
        mlflow.log_param("scale_features", config["model"]["scale_features"])
        mlflow.log_param("features_dropped", str(config["model"]["features_to_drop"]))

        # Log model-specific hyperparameters based on model type
        if config["model"]["model_type"] == "logistic_regression":
            mlflow.log_param("C", config["model"]["lr_C"])
        elif config["model"]["model_type"] == "random_forest":
            mlflow.log_param("n_estimators", config["model"]["rf_n_estimators"])
            mlflow.log_param("max_depth", str(config["model"]["rf_max_depth"]))
        elif config["model"]["model_type"] == "gradient_boosting":
            mlflow.log_param("n_estimators", config["model"]["gb_n_estimators"])
            mlflow.log_param("learning_rate", config["model"]["gb_learning_rate"])
            mlflow.log_param("max_depth", config["model"]["gb_max_depth"])

        # ── Load and prepare data ──
        features, target, n_rows, numeric_cols, categorical_cols = load_and_prepare_data(config)

        mlflow.log_param("n_rows", n_rows)
        mlflow.log_param("n_features", features.shape[1])

        # ── Split the data into training and test sets ──
        features_train, features_test, target_train, target_test = train_test_split(
            features, target,
            test_size=config["model"]["test_size"],
            random_state=config["model"]["random_state"],
            stratify=target
        )

        # ── Optionally scale features ──
        if config["model"]["scale_features"]:
            scaler = StandardScaler()
            features_train = pd.DataFrame(scaler.fit_transform(features_train), columns=features_train.columns)
            features_test = pd.DataFrame(scaler.transform(features_test), columns=features_test.columns)

        # ── Train ──
        model = build_model(config)
        print(f"\nTraining {config['model']['model_type']}...")
        trained_model = model.fit(features_train, target_train)

        # ── Log the trained model as an artifact ──
        # Log the model securely by explicitly trusting the sklearn Tree type
        if config['model']['model_type'] == "logistic_regression":
            mlflow.sklearn.log_model(model, "model")
        elif config["model"]["model_type"] == "random_forest":
            ml.log_model(
                sk_model= model,
                name="random_forest",  # Replaced deprecated artifact_path with name
                skops_trusted_types=["sklearn.tree._tree.Tree"]  # Authorizes the tree structure
            )
        else:
            assert config["model"]["model_type"] == "gradient_boosting"
            ml.log_model(
                sk_model=model,
                name="gradient_boosting",  # Replaced deprecated artifact_path with name
                skops_trusted_types=["sklearn.tree._tree.Tree"]  # Authorizes the tree structure
            )

        print(f"Finished training {config['model']['model_type']}.")
    return trained_model

if __name__ == "__main__":
    make_predictions()
