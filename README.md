# 📱 Mobile Price Range Predictor

A complete end-to-end machine learning project that predicts the price range of mobile phones based on their specifications. The project uses a Random Forest Classifier trained on a dataset of mobile phone features and is deployed as an interactive web application using Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_STREAMLIT_APP_URL_HERE) <-- *Replace with your app's URL after deploying!*

---

## 🚀 Project Overview

The goal of this project is to build a system that can accurately classify mobile phones into four price categories:

-   **0: Low Cost** (Under ₹10,000)
-   **1: Medium Cost** (₹10,000 - ₹20,000)
-   **2: High Cost** (₹20,000 - ₹40,000)
-   **3: Very High Cost** (Above ₹40,000)

This is achieved by training a machine learning model on key features like RAM, battery power, camera quality, and processing speed. The final model is then served through a user-friendly web interface.

---

## ✨ Key Features

-   **Interactive Web Interface:** A clean and simple UI built with Streamlit to get instant predictions.
-   **Machine Learning Model:** Utilizes a `RandomForestClassifier` from Scikit-learn, achieving an accuracy of ~88% on the test set.
-   **Data Preprocessing:** Implements `StandardScaler` to normalize features for optimal model performance.
-   **End-to-End Workflow:** Covers every step from data loading and cleaning to model training, evaluation, and deployment.

---

## 🛠️ Tech Stack

-   **Programming Language:** Python
-   **Machine Learning Libraries:** Scikit-learn, Pandas, NumPy
-   **Web Framework:** Streamlit
-   **Model Persistence:** Joblib

---

## 🔧 How to Run This Project Locally

Follow these steps to set up and run the project on your own machine.

**1. Clone the repository:**
```bash
git clone [https://github.com/YOUR_USERNAME/streamlit-phone-predictor.git](https://github.com/YOUR_USERNAME/streamlit-phone-predictor.git)
cd streamlit-phone-predictor