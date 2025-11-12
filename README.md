# Loan Approval Prediction

This notebook demonstrates a machine learning workflow to predict loan approval based on various applicant features.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [Data Preprocessing](#data-preprocessing)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Model Comparison](#model-comparison)
- [Decision Tree Rules](#decision-tree-rules)
- [Principal Component Analysis Visualization](#principal-component-analysis-visualization)

## Project Overview

The goal of this project is to build and compare two classification models, Decision Tree and Naive Bayes, to predict whether a loan application will be approved. The notebook covers data loading, preprocessing (handling missing values, one-hot encoding, scaling), dimensionality reduction using PCA, model training, and evaluation.

## Data Source

The data is loaded from a CSV file named `loan_approval.csv`. The dataset contains information about loan applicants, including:

- `name`: Applicant's name
- `city`: Applicant's city
- `income`: Applicant's income
- `credit_score`: Applicant's credit score
- `loan_amount`: Requested loan amount
- `years_employed`: Number of years employed
- `points`: Applicant's points (presumably a scoring metric)
- `loan_approved`: Target variable (True/False)

## Data Preprocessing

The following data preprocessing steps are performed:

1. **Handling Missing Values**: Missing values are checked and rows with missing values are dropped.
2. **One-Hot Encoding**: Categorical columns (`name` and `city`) are converted into numerical format using one-hot encoding.
3. **Feature Scaling**: Numerical features (`income`, `credit_score`, `loan_amount`, `years_employed`, `points`) are scaled using `StandardScaler`.

## Dimensionality Reduction

Principal Component Analysis (PCA) is applied to reduce the dimensionality of the dataset while retaining 95% of the variance.

## Model Training

Two classification models are trained:

1. **Decision Tree Classifier**: A Decision Tree model is trained on the PCA-transformed training data.
2. **Naive Bayes Classifier**: A Gaussian Naive Bayes model is trained on the PCA-transformed training data.

## Model Evaluation

Both models are evaluated on the test set using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Model Comparison

The performance metrics of the Decision Tree and Naive Bayes models are compared to determine which model performs better on this dataset.

Based on the evaluation results, the Decision Tree model achieved a significantly higher accuracy (0.9925) and better performance across all metrics compared to the Naive Bayes model (Accuracy: 0.62).

## Decision Tree Rules

The rules extracted from the trained Decision Tree model are printed, providing insights into the decision-making process of the model.

## Principal Component Analysis Visualization

Histograms are generated to visualize the distribution of selected principal components for both loan approval statuses (Approved and Not Approved). This helps understand how the principal components separate the two classes.
