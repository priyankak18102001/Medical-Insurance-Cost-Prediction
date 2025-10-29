# 🏥 Medical Insurance Cost Prediction

## 1️ Project Overview
This project focuses on predicting individual medical insurance costs based on demographic and lifestyle factors such as age, BMI, smoking habits, region, and number of children.

It combines data cleaning, feature engineering, regression modeling, and MLflow integration for experiment tracking.
A user-friendly Streamlit web app is also built for interactive cost prediction and visualization.

## 2 Objective
Goal: Predict the medical insurance cost (charges) for an individual using regression models.
Business Use Case: Helps insurance companies estimate premiums fairly and identify key cost drivers.

Deliverables:
Python scripts for data cleaning, feature engineering, and model training
A cleaned dataset
Regression models tracked via MLflow
Streamlit web app for predictions and EDA
Full documentation and insights (this README)

## 3 Dataset Description
Source: Kaggle — Medical Cost Personal Dataset
Rows: 1338 Columns: 7
Feature	 :Description
age	 :Age of the primary beneficiary
sex	 :Gender (male/female)
bmi	 :Body Mass Index — indicator of weight relative to height
children :	Number of dependents covered by insurance
smoker :	Smoking status (yes/no)
region :	Residential area in the US
charges :	Medical insurance cost (target variable)

## 4 Data Cleaning Process
Removed duplicate and invalid entries

Dropped rows with missing values in key columns

Standardized text columns (sex, smoker, region)

Filtered out unrealistic BMI (<10 or >70) and ages (<18 or >120)

Ensured numeric data types for modeling

## 5 Feature Engineering
Created a bmi_category feature: underweight, normal, overweight, obese

Encoded categorical features with One-Hot Encoding

Standardized numerical features (Age, BMI, Children)

Added interaction terms in Streamlit app for better model interpretation:

smoker_age = smoker * age

smoker_bmi = smoker * bmi

## 6 Modeling & MLflow Integration
Trained multiple regression models using scripts/train_model_mlflow.py:

Ridge Regression

XGBoost Regressor

Best model selected based on lowest RMSE on test data.
All experiments, metrics, and parameters were tracked using MLflow.

MLflow Setup:

mlflow ui

Accessible at: http://127.0.0.1:5000

## 7 Evaluation Metrics & Results
Metric	Ridge Regression	XGBoost
RMSE	4544.33	5004.59
MAE	2809.45	2804.16
R²	0.8876	R2=0.8637
Selected Model: Ride (registered as MedicalInsuranceCostModel in MLflow)

## 8 Streamlit Web Application
Features:

Interactive form to input personal data (age, BMI, smoker, etc.)

Visual EDA plots (Age vs Charges, BMI vs Charges, Region comparison)

Real-time prediction using the trained model

Optional 95% Confidence Interval based on RMSE or fallback error %

## 9 Business Insights & Recommendations
Key Insights:

Smoking status is the most significant cost driver — smokers pay ~3x higher premiums.

BMI and age are positively correlated with insurance charges.

Regional variation is relatively minor compared to lifestyle factors.

Recommendations:

Encourage wellness programs targeting BMI reduction and smoking cessation.

Offer age-based premium segmentation to balance fairness and profitability.

Monitor high-risk groups with predictive analytics for preventive care incentives.
