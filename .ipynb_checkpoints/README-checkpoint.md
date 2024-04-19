# Heart Disease Prediction Project

## Introduction
This project utilizes the UCI Heart Disease dataset to build machine learning models that predict the presence of heart disease in patients based on various medical and demographic features. This tool aims to aid in early diagnosis and facilitate preventative or remedial actions by medical professionals.

## Dataset
The data for this project comes from the UCI Machine Learning Repository and can be accessed [here](https://archive.ics.uci.edu/ml/datasets/Heart+Disease). It includes various features such as blood pressure, cholesterol levels, heart rate, and other cardiovascular conditions.

## Setup
To run this project, ensure you have the following tools installed:
- Python 3
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

You can install the necessary libraries using:
bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn

## Usage
1) Data Preparation: The initial raw dataset undergoes several preprocessing steps:
    - Missing values are handled.
    - Outlier detection and replacement.
    - Feature encoding and scaling.
2) Exploratory Data Analysis (EDA): This includes visualizations for both univariate and multivariate analysis to understand the distributions and relationships in the data.
3) Model Building: We use several machine learning models to predict heart disease:
    - Random Forest Classifier
    - Logistic Regression
    - Decision Tree Classifier
    - XGBoost Classifier
4) Evaluation: Models are evaluated based on their accuracy, precision, recall, and F1-score to understand their effectiveness.

## Running the Project
Navigate to the project directory and run the script file:

bash
Copy code
python heart_disease_prediction.py
Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change. Ensure to update tests as appropriate.

License
Distributed under the MIT License. See LICENSE for more information.

Acknowledgements
UCI Machine Learning Repository for providing the dataset.
Scikit-learn and XGBoost communities for excellent documentation and APIs.