# SC1015 - Heart Disease Prediction Project

## About
This is a Mini-Project for SC1015 - Introduction to Data Science and Artificial Intelligence (DSAI) which focuses on predicting the presence of heart disease in patients based on various medical and demographic features from UCI Heart Disease dataset.

## Contributors

- **Jiang Li Kai** - Data Cleaning & Preparation, EDA, Machine Learning
- **Heng Zeng Xi** - Machine Learning, Data-Driven Insights, Presentation, Readme
- **Ang Wei Ming** - Reasoning For Data Cleaning, Insights on EDA, Slides



## Motivation
1. Cardiovascular diseases, including heart disease, account for approximately 18 million deaths globally each year, early detection is crucial for effective intervention. Source from https://ourworldindata.org/cardiovascular-diseases#:~:text=As%20you%20can%20see%2C%20heart,total%20of%20around%2018%20million.
2. By leveraging machine learning models, healthcare providers can identify high-risk individuals and implement targeted prevention and treatment strategies, reducing the overall burden of heart disease on healthcare systems.
   Source from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10417090/

## Dataset
The data for this project comes from the UCI Machine Learning Repository and can be accessed [here](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

## Problem definition
* Can we accurately predict the presence of heart disease based on medical and demographic features?
* Which features have the most significant impact on predicting heart disease risk?

## 1. Data Cleaning & Preparation

In this project phase, we meticulously prepared and cleaned the dataset to optimize data analysis and facilitate subsequent machine learning tasks. However, a major issue was the incompleteness and inconsistency of the data.

To solve this, we performed the following:

1. **Handling Missing Values**:
   - For numerical features ('trestbps', 'chol', 'thalch', 'oldpeak', 'ca'), missing values were imputed using the median strategy.
   - For categorical features ('sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'), missing values were imputed using the most frequent strategy.

2. **Outlier Detection and Handling**:
   - Outliers in numerical features were identified using the interquartile range (IQR) method.
   - Outliers were replaced with boundary values to ensure data integrity.

3. **Feature Encoding**:
   - Categorical features were one-hot encoded to convert them into numerical format for machine learning algorithms.

4. **Data Standardization/Normalization**:
   - Numerical features were standardized using the StandardScaler to ensure all features have a mean of 0 and standard deviation of 1.

## 2. Exploratory Data Analysis

We engaged in Exploratory Data Analysis (EDA) to uncover patterns, examine the distributions of success, and explore potential relationships between them. This step was crucial for gaining insights and informing our research question.

To achieve this, we did the following:

1. **Univariate Analysis of Numerical Features**:
   - Explored distributions of numerical features (age, trestbps, chol, thalach, oldpeak). Visualized distributions using histograms with KDE (Kernel Density Estimation).
   
2. **Distribution of Categorical Features**:
   - Visualized distributions of categorical features (sex, cp, fbs, restecg, exang, slope, thal). Used countplots to show the frequency of each category.

3. **Multivariate Analysis**:
   - Examined relationships among numerical features and their connection with the target variable using a pairplot.
   - Explored an in-depth analysis of the relationship between categorical characteristics and the occurrence of heart disease using a percent-stacked bar chart.

4. **Correlation Analysis**
   
5. **Chi-Squared Test**:
   - Applied chi-squared tests for each variable against the target variable 'num_category' to evaluate the significance of their association.
     - For age group, the chi-squared statistic was 143.72 with a p-value of approximately 1.06e^-24, indicating a significant association with heart disease occurrence.
     - The chi-squared test for cholesterol group yielded a statistic of 39.81 and a p-value of approximately 3.47e^-06, indicating a significant association with heart disease.

6. **Analysis of Variance (ANOVA)**:
   - Conducted separate ANOVA tests for each variable to examine their influence on heart disease occurrence.
     - For age, the ANOVA test yielded an F-statistic of 31.23 with a p-value of approximately 2.11e^-24, indicating a significant association with heart disease.
     - Thalach exhibited an F-statistic of 42.15 and a p-value of approximately 1.80e^-32, indicating a significant relationship with heart disease occurrence.

## 3. Machine Learning Techniques

In this phase, we employed a variety of machine learning techniques to develop robust predictive models for heart disease detection.

To achieve this, we did the following:

1. **Principal Component Analysis (PCA)**:
   - Applied preprocessing steps to scale numerical features and encode categorical features. Visualized the first two principal components to gain insights into the data's variance and distribution. There is overlap between the two groups, regions with larger principal component 1 tend to have a higher concentration of individuals with heart disease.

2. **Random Forest Classifier**:
   - Split the dataset into training and testing sets to evaluate the model's accuracy and generalization. Performed hyperparameter tuning using grid search to optimize the model's performance. When evaluated on the test set, the model demonstrated exceptional accuracy, achieving a score of 1.0. The classification report indicates perfect precision, recall, and F1-scores for both classes, with a macro and weighted average F1-score of 1.0.

3. **Learning Curve**:
   - The learning curve provides valuable insights into how the model's performance evolves as it learns from increasingly larger portions of the training data. It also helps to identify potential issues such as overfitting or underfitting.

4. **XGBoost Classifier**:
   - Trained an XGBoost model to classify instances of heart disease.

5. **Employed SHAP (SHapley Additive exPlanations)**:
   - To explain the model's predictions and visualize feature importance. Identified the top important features contributing to the model's predictions. Top 5 important features: cp_asymptomatic, chol, oldpeak, sex_Female, thalch.

6. **Decision Tree Classifier**:
   - Utilized a Decision Tree Classifier to classify instances of heart disease using only the top 5 important features. Evaluated the model's accuracy with the selected features. The decision tree model trained on the top 5 important features correctly predicts the presence or absence of heart disease in approximately 80.9% of the test samples.

7. **Logistic Regression**:
   - Trained a Logistic Regression model on the top 5 important features to classify instances of heart disease. Calculated the accuracy of the model on the test data. An accuracy of 0.788 means that the Logistic Regression model correctly predicted the presence or absence of heart disease in approximately 78.8% of the samples in the test dataset.
   - ROC AUC of 0.87, the logistic regression model demonstrates strong discriminative power. This indicates that the model is effective in correctly ranking the predicted probabilities of heart disease, with a higher probability assigned to individuals who actually have heart disease compared to those who do not


    


## 4. Conclusion
  In conclusion, our analysis aimed to predict heart disease occurrence and understand the factors influencing it. We meticulously cleaned and prepared the data, addressing issues like missing values and outliers. Exploratory Data Analysis revealed significant 
  associations between age, cholesterol levels, and heart disease.

  Machine learning models, including PCA and Random Forest Classifier, demonstrated high accuracy in predicting heart disease. The learning curve analysis provided insights into model performance. SHAP values highlighted key features contributing to predictions.
 
  Notably, the Chi-Squared Test and ANOVA unveiled striking associations between age, cholesterol levels, and heart disease occurrence. With chi-squared statistics of 143.72 and 39.81, and p-values of approximately 1.06e^-24 and 3.47e^-06, respectively, these tests 
  underscored the importance of age-related factors and cardiovascular health indicators in predicting heart disease.

  Overall, our analysis offers valuable insights into heart disease prediction, emphasizing the importance of early detection and intervention strategies.



## References

- [Our World in Data - Cardiovascular Diseases](https://ourworldindata.org/cardiovascular-diseases#:~:text=As%20you%20can%20see%2C%20heart,total%20of%20around%2018%20million.)
- [NCBI - Cardiovascular Disease Prediction Using Machine Learning Techniques](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10417090/)
- [Towards Data Science - Logistic Regression and Decision Boundary](https://towardsdatascience.com/logistic-regression-and-decision-boundary-eab6e00c1e8)
- [Investopedia - Chi-Square Statistic](https://www.investopedia.com/terms/c/chi-square-statistic.asp)
- [Analytics Vidhya - ANOVA (Analysis of Variance)](https://www.analyticsvidhya.com/blog/2018/01/anova-analysis-of-variance/)
- [Machine Learning Mastery - Learning Curves for Diagnosing Machine Learning Model Performance](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)
- [Analytics Vidhya - Understanding XGBoost](https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/)
- [Towards Data Science - Using SHAP Values to Explain How Your Machine Learning Model Works](https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137)
- [Scikit-learn - Grid Search](https://scikit-learn.org/stable/modules/grid_search.html)

