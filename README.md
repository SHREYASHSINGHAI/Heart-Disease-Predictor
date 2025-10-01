# Heart-Disease-Predictor
This project develops a predictive model for Heart Disease, leveraging a comprehensive machine learning pipeline. It begins with Data Preparation, where raw data is transformed; numerical features are retained, and categorical variables are converted using One-Hot Encoding.

The project explores three classification algorithms: Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest. These models are initially benchmarked for baseline performance.

The heart of the project lies in Model Optimization through hyperparameter tuning. GridSearchCV is applied to KNN for fine-tuning parameters like n_neighbors and distance metrics. RandomizedSearchCV is utilized for both Random Forest (optimizing n_estimators, max_depth) and Logistic Regression (tuning the regularization strength C), efficiently searching for the best configurations.

The optimized models undergo thorough Evaluation. This includes visual assessments via ROC Curves and Confusion Matrices, detailed performance insights from Classification Reports, and robust validation using 5-fold Cross-Validation to calculate mean Accuracy, Precision, Recall, and F1-score.

Finally, the project concludes with Feature Importance Analysis, extracting coefficients from the best Logistic Regression model to highlight key factors influencing heart disease prediction. This end-to-end pipeline ensures a robust and interpretable predictive solution.
