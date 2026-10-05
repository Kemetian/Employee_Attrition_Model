# Employee_Attrition_Model
Classification model predicting whether employees leave, with a good mix of feature types.

Dataset used: IBM HR Analytics Employee Attrition CSV

Preprocessing: This dataset has been preprocessed. There are no missing values and all numerical features have been encoded. The experiment.py script enacts label encoding to encode the categorical columns. The encoding key is located in the data folder as encoding_key.

Operation instructions: Set desired configuration(s) in src/sweep.py. Run sweep.py to train and measure each configuration. Run analyze.py to view top 5 performing model configurations.

# Data Drift Analysis Report (Sprint 17)

1. Identified Drifted Features
    Our automated monitoring pipeline utilizing **Evidently AI** flagged statistically significant data drift (using Kolmogorov-Smirnov and Chi-Square tests) across the following key features:
        **`average_monthly_hours`**: The distribution of employee work hours has shifted significantly compared to the baseline tracking dataset.
        **`satisfaction_level`**: Self-reported employee satisfaction scores show a structural shift in distribution, changing core density bands.

2. Expected Impact on Model Performance
    Because **`satisfaction_level`** and **`average_montly_hours`** are highly weighted features in our `RandomForestClassifier` configuration, distribution changes pose an immediate risk:
        **Feature Importance Distortion**: The decision splits created during the baseline training run may no longer align with current distribution boundaries.
        **Performance Degradation**: We anticipate a drop in macro F1-score and accuracy metrics due to **data leakage/concept mismatch** as real-world trends deviate from what the model originally mapped.
        **Unreliable Feature Weights**: Prediction confidence intervals will widen, making automated downstream classifications unstable.

3. Recommended Remediation Actions
    To restore statistical integrity and pipeline accuracy, we recommend executing the following prioritized steps:
        1. **Trigger Automated Re-training**: Re-run `src/experiment.py` utilizing an expanded training window that includes the newly acquired DVC-tracked dataset to capture current distribution variances.
        2. **Review Feature Preprocessing Bounds**: Inspect outlier handling and binning transformations for structural skew within `src/evaluation.py`.
        3. **Establish Alert Thresholds**: Wire the `DriftReport` flags directly into our MLflow runtime to automatically stop automated deployments if a dataset-level drift threshold (p-value < 0.05) is breached.
