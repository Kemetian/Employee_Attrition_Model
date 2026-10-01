# Employee_Attrition_Model
Classification model predicting whether employees leave, with a good mix of feature types.

Dataset used: IBM HR Analytics Employee Attrition CSV

Preprocessing: This dataset has been preprocessed. There are no missing values and all numerical features have been encoded. The experiment.py script enacts label encoding to encode the categorical columns. The encoding key is located in the data folder as encoding_key.

Operation instructions: Set desired configuration(s) in src/sweep.py. Run sweep.py to train and measure each configuration. Run analyze.py to view top 5 performing model configurations.
