# Online Shoppers Purchasing Intention Predictor

An end-to-end machine learning web application built with Streamlit and XGBoost to predict whether an online website visitor will complete an e-commerce transaction based on session metrics and user behavior.

## Features
* **Interactive Web Interface:** Input real-time session variables (such as page values, duration, and bounce rates) via a clean Streamlit dashboard.
* **Optimized ML Pipeline:** Utilizes a tuned XGBoost classifier trained to detect high-conversion shopping sessions.
* **Detailed Documentation:** Includes comprehensive breakdowns of all input metrics and categorical features.

## Tech Stack
* **Python**
* **Streamlit** (Web framework)
* **XGBoost & Scikit-Learn** (Modeling & evaluation)
* **Pandas & NumPy** (Data processing)
* **Joblib** (Artifact serialization)

## Project Structure
```text
├── app.py                        # Main Streamlit application script
├── main.ipynb                    # Training and data science experimentation notebook
├── xgb_model.json                # Serialized XGBoost model artifact
├── model_features.pkl            # Saved feature column mapping
├── online_shoppers_intention.csv # Source dataset
├── feature_explanations.txt      # Reference guide for input variables
└── requirements.txt              # Project dependencies
