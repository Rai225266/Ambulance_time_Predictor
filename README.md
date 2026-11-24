# 🚑 AI-Based Ambulance Response Time & Delay Risk Predictor

## 📌 Project Overview
The **AI-Based Ambulance Response Time & Delay Risk Predictor** is a Machine Learning project designed to estimate ambulance travel time and identify the likelihood of delays during emergency responses. It uses emergency location, distance, traffic conditions, weather, road conditions, and time of day to generate predictions through an interactive Streamlit dashboard.

The system provides:
- **Predicted Response Time:** Estimated ambulance travel time in minutes.
- **Delay Probability:** Likelihood of the ambulance being delayed.
- **Risk Category:** Low, Moderate, High, or Critical.
- **Expected Delay:** Estimated additional delay, where sufficient data is available.

## 🎯 Problem Statement
Ambulance delays can affect emergency medical outcomes. This project explores how Machine Learning can help identify delay patterns and support better emergency response planning by analysing dispatch-related conditions.

## 📊 Dataset
The project uses an EMS incident dispatch dataset containing **1,000 records and 12 columns**.

Key features include:
- Distance and Average Speed
- Traffic Level and Weather
- Time of Day and Day of Week
- Road Type and Emergency Type
- Traffic Signals and Rainfall
- Hospital Distance
- On_Time (Target Variable)

The dataset contains approximately 5% missing values per feature and a relatively balanced target distribution.

## ⚙️ Technologies Used
- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Joblib
- Streamlit

## 🧠 Machine Learning Approach

**Data Preprocessing:** Missing-value imputation, duplicate removal, outlier handling, categorical encoding, feature scaling, and stratified train-test splitting.

**Exploratory Data Analysis:** Histograms, boxplots, correlation heatmaps, and feature analysis to identify relationships between dispatch conditions and delays.

**Regression Models:**
- Linear Regression
- Random Forest Regressor

**Classification Models:**
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest Classifier

Models are compared using cross-validation and appropriate evaluation metrics, including Accuracy, Precision, Recall, F1-score, MAE, RMSE, and R².

## 📈 Key Results
- Approximately **97% classification accuracy** when all dataset features, including average speed, are available.
- Approximately **65% accuracy** using the dashboard's five primary input features.
- Identified distance and average speed as important predictors of ambulance delays.

## ⚠️ Limitations
Actual response-time records are unavailable. Therefore, travel time is estimated using distance and average speed. The dataset also appears synthetic, and its category codes require assumptions.

**This project is an academic prototype, not a validated system for real-world emergency dispatch decisions.**

## 🚀 Future Scope
- Integration with Google Maps or live traffic APIs.
- Real-time weather data integration.
- Collection of actual ambulance response records.
- Improved prediction accuracy using larger real-world datasets.
- GPS-based ambulance tracking and route optimisation.

## 👨‍💻 Developed By
**Sourabh Rai**  
B.Tech – Computer Science & Engineering (AI)  
Shoolini University

**Project Domain:** Machine Learning | Artificial Intelligence | Healthcare Technology
