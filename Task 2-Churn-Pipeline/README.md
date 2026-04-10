#  Task 2: End-to-End Customer Churn Pipeline

### Objective
To develop a production-ready machine learning system that predicts the likelihood of customer attrition (churn) using the IBM Telco Dataset. The primary focus of this task is the implementation of a reusable, automated pipeline using the Scikit-Learn Pipeline API to ensure data consistency, prevent data leakage, and streamline model deployment.

###  Tech Stack & Methodology
This project utilizes **Python**, **Pandas**, and **Scikit-Learn** to transform raw data into actionable insights. To handle the mixed nature of the dataset, a `ColumnTransformer` was implemented: **Numeric Features** (Tenure, MonthlyCharges, TotalCharges) were processed using a `SimpleImputer` and `StandardScaler`, while **Categorical Features** were handled via a `SimpleImputer` and `OneHotEncoder`. The final model consists of a **Random Forest Classifier** optimized through **GridSearchCV** over parameters like `n_estimators` and `max_depth`. The model was evaluated using a **Weighted F1-Score (0.78)** and **Accuracy (0.79)**, ensuring a balanced performance on an imbalanced dataset.

###  Key Insights & Visualizations
Beyond pure metrics, the project includes high-level analytical visualizations to provide business value. An **Interactive 3D Scatter Plot** (Tenure vs. Monthly Charges vs. Total Charges) was developed to visualize high-risk customer clusters. Additionally, a **Confusion Matrix** and **Precision-Recall Curve** were utilized to assess the model's reliability in identifying churners. Feature importance analysis revealed that **Contract Type** and **Tenure** are the strongest predictors of customer retention, providing clear direction for business strategy.

###  Production-Ready Export
The complete architecture—including all preprocessing steps and the tuned classifier—has been serialized into a single file: `churn_pipeline_v1.joblib`. This allows for seamless integration into production environments, where raw, uncleaned data can be fed directly into the pipeline for instant predictions without manual data engineering.

###  Skills Gained
- **ML Pipeline Construction:** mastering the `Pipeline` and `ColumnTransformer` API for leak-proof workflows.
- **Hyperparameter Tuning:** systematic optimization of models using `GridSearchCV`.
- **Model Serialization:** exporting complete artifacts for real-world deployment using `joblib`.
- **Business Logic:** translating technical metrics into churn prevention insights.

- ## Author:
- Khadija Abdulrahman
