from configs.model_config import config as base_config
from evaluate import make_predictions

# Define a list of experiments to try
experiments = [
    {
        "model_type": "logistic_regression",
        "lr_C": 0.01,
    },
    {
        "model_type": "logistic_regression",
        "lr_C": 0.1,
    },
    {
        "model_type": "logistic_regression",
        "lr_C": 10.0,
    },
    {
        "model_type": "random_forest",
        "rf_n_estimators": 30,
        "rf_max_depth": 3,
    },
    {
        "model_type": "random_forest",
        "rf_n_estimators": 50,
        "rf_max_depth": 3,
    },
    {
        "model_type": "gradient_boosting",
        "gb_n_estimators": 30,
        "gb_learning_rate": 0.01,
        "gb_max_depth": 3,
    },
    {
        "model_type": "gradient_boosting",
        "gb_n_estimators": 50,
        "gb_learning_rate": 0.1,
        "gb_max_depth": 3,
    },
]

print(f"Running {len(experiments)} experiments...\n")

for i, overrides in enumerate(experiments):
    print(f"\n{'='*60}")
    print(f"Experiment {i+1}/{len(experiments)}")
    print(f"{'='*60}")

    # Start with the base config and apply overrides
    current_config = base_config.copy()
    current_config.update(overrides)

    try:
        run_id = make_predictions(current_config)
        print(f"Completed. Run ID: {run_id}")
    except Exception as e:
        print(f"Failed: {e}")

print(f"\nAll experiments complete. Run 'mlflow ui' to compare results.")
