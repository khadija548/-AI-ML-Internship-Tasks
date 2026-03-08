# Task 3: Heart Disease Prediction & Advanced EDA 

# Project Objective
The goal of this project was to build a machine learning model to predict whether a person is at risk of heart disease based on their clinical health data (UCI Cleveland Dataset). 

##  Key Requirements Met
- **Data Cleaning:** Handled missing values represented by `?`, performed type conversion, and renamed the target variable for clarity.
- **Exploratory Data Analysis (EDA):** Performed advanced **4D visualization** to analyze the relationship between Age, Max Heart Rate, ST Depression, and Diagnosis.
- **Classification Modeling:** Trained a model to categorize risk levels.
- **Evaluation:** Analyzed performance using Accuracy, Confusion Matrix, and ROC-AUC curves.
- **Feature Importance:** Identified the clinical indicators most responsible for predicting heart disease.

##  Model Performance
The model achieved strong predictive power, allowing for reliable classification:
- **ROC-AUC Score:** 0.90
- **Top 3 Risk Factors:**
  1. **ca (Major Vessels):** 1.14 Importance
  2. **sex:** 0.64 Importance
  3. **thal (Thalassemia):** 0.53 Importance



##  Medical Insights
- **Vascular Health:** The number of major vessels (`ca`) is the strongest predictor of disease in this dataset.
- **Age & Stress:** 4D analysis confirms that the highest risk occurs in patients over 55 who exhibit ST depression and a suppressed maximum heart rate (`thalach`) during exercise.


1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Open `Heart_Disease_Analysis.ipynb` in Jupyter Notebook or Google Colab.
