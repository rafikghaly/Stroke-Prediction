# Stroke Prediction Data Mining Project

## Overview
This project aims to predict stroke occurrences using machine learning models and clustering algorithms. The dataset is preprocessed to enhance model accuracy, and various algorithms are implemented to identify factors contributing to stroke risks.

## Objective
The primary goal is to develop effective models for predicting strokes based on demographic, lifestyle, and health-related factors.

## Dataset
- **Features**: gender,	age,	hypertension,	heart_disease,	ever_married,	work_type,	Residence_type,	avg_glucose_level,	bmi,	smoking_status,	stroke.
- **Target Variable**: Stroke (1 if the patient had a stroke, 0 otherwise)

## Models and Algorithms Used
- **MLP Classifier**: Multi-layer Perceptron for neural network-based classification.
- **Naive Bayes Classifier**: Probabilistic model based on Bayes' theorem.
- **Agglomerative Clustering**: Hierarchical clustering method.
- **K-Nearest Neighbors (KNN)**: Instance-based learning for classification.
- **DBSCAN Clustering**: Density-based spatial clustering of applications with noise (unsupervised).
- **K-Means Clustering**: Centroid-based clustering (unsupervised).
- **Support Vector Machines (SVM)**: Linear and nonlinear models for classification.
- **Decision Tree**: Tree-based model for classification.
- **Logistic Regression**: Statistical model for binary classification.

## Methodology
1. **Preprocessing**:
   - **Data Cleaning**: Handling missing values, outliers, and inconsistent data.
   - **Feature Selection**: Selecting relevant features using correlation analysis and domain knowledge.
   - **Normalization/Scaling**: Scaling numerical features to ensure consistent model performance.
   - **Encoding**: Encoding categorical variables using techniques like one-hot encoding.
   - **Train-Test Split**: Splitting the dataset into training and testing sets.

2. **Model Training and Evaluation**:
   - Training each model on the training set.
   - Evaluating model performance using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.

3. **Cluster Analysis**:
   - Applying clustering algorithms to identify patterns and segments related to stroke risk.

## Results
- **Model Comparison**: Comparison of model performance and effectiveness.
- **Insights from Clustering**: Discovering clusters related to different stroke risk profiles.
- **Feature Importance**: Determining key features influencing stroke prediction.

## Future Work
- **Feature Engineering**: Exploring additional features or transformations.
- **Hyperparameter Tuning**: Optimizing model parameters for better performance.
- **Enhanced Dataset**: Incorporating more diverse and larger datasets for robust models.
- **Deployment**: Developing a user-friendly interface or API for real-time stroke risk prediction.
