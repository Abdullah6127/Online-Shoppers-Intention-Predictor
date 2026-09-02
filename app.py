import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier
from pathlib import Path

# Set page layout and title
st.set_page_config(
    page_title="Online Shopper Purchase Intention Predictor",
    page_icon="🛍️",
    layout="wide"
)

# Load model and model artifacts
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    model_path = BASE_DIR / "xgb_model.json"
    features_path = BASE_DIR / "model_features.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")

    model = XGBClassifier()
    model.load_model(str(model_path))

    model_features = joblib.load(features_path)

    return model, model_features

try:
    model, model_features = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

st.title("🛍️ Online Shopper Purchase Intention Predictor")
st.write("Predict whether a website visitor will complete a transaction (`Revenue = True`).")

# Form layout using tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Page Activity & Duration", 
    "📈 Page Metrics", 
    "💻 System & Traffic", 
    "📅 Session & Visitor Info"
])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        admin = st.number_input("Administrative Page Views", min_value=0, max_value=50, value=2)
        admin_duration = st.number_input("Administrative Duration (s)", min_value=0.0, value=80.0)
        info = st.number_input("Informational Page Views", min_value=0, max_value=50, value=0)
        info_duration = st.number_input("Informational Duration (s)", min_value=0.0, value=0.0)
    with col2:
        product = st.number_input("Product Related Page Views", min_value=0, max_value=1000, value=30)
        product_duration = st.number_input("Product Related Duration (s)", min_value=0.0, value=1100.0)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        bounce_rate = st.slider("Bounce Rate", min_value=0.0, max_value=0.2, value=0.01, step=0.001)
        exit_rate = st.slider("Exit Rate", min_value=0.0, max_value=0.2, value=0.03, step=0.001)
    with col2:
        page_values = st.number_input("Page Value", min_value=0.0, max_value=400.0, value=6.0)
        special_day = st.selectbox("Special Day Closeness", options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], index=0)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        operating_systems = st.selectbox("Operating System ID", options=list(range(1, 9)), index=1)
        browser = st.selectbox("Browser ID", options=list(range(1, 14)), index=1)
    with col2:
        region = st.selectbox("Region ID", options=list(range(1, 10)), index=2)
        traffic_type = st.selectbox("Traffic Type ID", options=list(range(1, 21)), index=1)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        month = st.selectbox("Month", options=['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], index=2)
        visitor_type = st.selectbox("Visitor Type", options=['Returning_Visitor', 'New_Visitor', 'Other'], index=0)
    with col2:
        weekend = st.selectbox("Weekend Visit", options=[False, True], index=0)

st.markdown("---")

# Predict button
if st.button("🚀 Predict Purchase Intention", type="primary"):
    # Build raw input dataframe
    raw_data = {
        'Administrative': [admin],
        'Administrative_Duration': [admin_duration],
        'Informational': [info],
        'Informational_Duration': [info_duration],
        'ProductRelated': [product],
        'ProductRelated_Duration': [product_duration],
        'BounceRates': [bounce_rate],
        'ExitRates': [exit_rate],
        'PageValues': [page_values],
        'SpecialDay': [special_day],
        'Month': [month],
        'OperatingSystems': [operating_systems],
        'Browser': [browser],
        'Region': [region],
        'TrafficType': [traffic_type],
        'VisitorType': [visitor_type],
        'Weekend': [weekend]
    }
    
    input_df = pd.DataFrame(raw_data)

    # Perform One-Hot Encoding to match training preprocessing
    encoded_df = pd.get_dummies(input_df, columns=['Month', 'VisitorType'], drop_first=False)
    encoded_df['Weekend'] = encoded_df['Weekend'].astype(int)

    # Reindex columns to guarantee exact 26 feature order expected by the model
    processed_df = encoded_df.reindex(columns=model_features, fill_value=0)

    # Make Prediction
    prediction = model.predict(processed_df)[0]
    
    # Calculate probability if model supports it
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(processed_df)[0][1]
    else:
        proba = None

    # Output Results
    st.subheader("Prediction Result")
    if prediction == 1 or prediction == True:
        st.success("✅ **Result: Customer WILL likely complete a purchase (Revenue = True)**")
        if proba is not None:
            st.info(f"Purchase Probability: **{proba * 100:.2f}%**")
    else:
        st.warning("❌ **Result: Customer WILL NOT likely complete a purchase (Revenue = False)**")
        if proba is not None:
            st.info(f"Purchase Probability: **{proba * 100:.2f}%**")