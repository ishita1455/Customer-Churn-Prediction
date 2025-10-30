import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🎯",
    layout="centered"
)

# =====================================================
# Custom CSS - Simple & Clean
# =====================================================
st.markdown("""
    <style>
    .big-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .result-high {
        background-color: #ff4444;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
    }
    .result-medium {
        background-color: #ffaa00;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
    }
    .result-low {
        background-color: #00C851;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================
# Load Models
# =====================================================
@st.cache_resource
def load_models():
    """Load trained model and preprocessors"""
    models_dir = Path("models_trained")
    
    try:
        with open(models_dir / "best_churn_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(models_dir / "preprocessors.pkl", "rb") as f:
            preprocessors = pickle.load(f)
        return model, preprocessors
    except FileNotFoundError:
        st.error("❌ Model files not found in 'models_trained' folder!")
        st.info("Please run the training script first.")
        st.stop()

model, preprocessors = load_models()
label_encoders = preprocessors["label_encoders"]
scaler = preprocessors["scaler"]
feature_names = preprocessors["feature_names"]

# =====================================================
# Header
# =====================================================
st.markdown('<h1 class="big-title">🎯 Customer Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter customer details to predict churn risk</p>', unsafe_allow_html=True)

# =====================================================
# Input Form
# =====================================================
st.markdown("### 📝 Enter Customer Information")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("**🛍️ Product Info**")
    product_category = st.selectbox(
        "Product Category",
        ["Electronics", "Clothing", "Home","Books", ]
    )
    
    product_price = st.number_input(
        "Product Price ($)",
        min_value=0.0,
        max_value=10000.0,
        value=50.0,
        step=5.0
    )
    
    quantity = st.number_input(
        "Quantity",
        min_value=1,
        max_value=100,
        value=1
    )
    
    total_purchase_amount = st.number_input(
        "Total Amount ($)",
        min_value=0.0,
        value=product_price * quantity,
        step=5.0
    )
    
    payment_method = st.selectbox(
        "Payment Method",
        ["Credit Card", "PayPal", "Cash"]
    )

with col2:
    st.markdown("**👤 Customer Info**")
    customer_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )
    
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )
    
    returns = st.radio(
        "Has returned items?",
        ["No", "Yes"],
        horizontal=True
    )
    returns_binary = 1 if returns == "Yes" else 0
    
    st.markdown("**📅 Purchase Date**")
    purchase_date = st.date_input(
        "Date",
        value=datetime.now()
    )

purchase_year = purchase_date.year
purchase_month = purchase_date.month
purchase_day = purchase_date.day

# =====================================================
# Prediction Function
# =====================================================
def predict_churn(input_data):
    """Make prediction"""
    df = pd.DataFrame([input_data])
    
    # Encode categorical features
    for col in ["Product Category", "Payment Method", "Gender"]:
        if col in label_encoders:
            try:
                df[col] = label_encoders[col].transform(df[col])
            except:
                df[col] = 0
    
    # Reorder to match training
    df = df[feature_names]
    
    # Scale ONLY the numeric columns (scaler was fit on only 5 numeric features)
    numeric_cols = ['Product Price', 'Quantity', 'Total Purchase Amount', 
                   'Customer Age', 'Returns']
    
    # Create a copy for scaling
    df_for_scaling = df[numeric_cols].copy()
    df_scaled_numeric = scaler.transform(df_for_scaling)
    
    # Replace numeric columns with scaled values
    df[numeric_cols] = df_scaled_numeric
    
    # Convert to numpy array
    df_array = df.values
    
    # Predict
    prediction = model.predict(df_array)[0]
    prediction_proba = model.predict_proba(df_array)[0]
    
    return prediction, prediction_proba

# =====================================================
# Predict Button
# =====================================================
st.markdown("---")

if st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary"):
    
    # Prepare input
    input_data = {
        "Product Category": product_category,
        "Product Price": product_price,
        "Quantity": quantity,
        "Total Purchase Amount": total_purchase_amount,
        "Payment Method": payment_method,
        "Customer Age": customer_age,
        "Returns": returns_binary,
        "Gender": gender,
        "purchase_year": purchase_year,
        "purchase_month": purchase_month,
        "purchase_day": purchase_day
    }
    
    # Make prediction
    with st.spinner("Analyzing..."):
        prediction, prediction_proba = predict_churn(input_data)
    
    # Calculate probabilities
    churn_prob = prediction_proba[1] * 100
    no_churn_prob = prediction_proba[0] * 100
    
    # Determine risk level
    if churn_prob >= 60:
        risk_level = "HIGH RISK"
        risk_class = "result-high"
        emoji = "🔴"
        message = "This customer is likely to churn!"
        action = "Action Required: Contact customer immediately with retention offers."
    
    else:
        risk_level = "LOW RISK"
        risk_class = "result-low"
        emoji = "🟢"
        message = "This customer is likely to stay!"
        action = "Keep it up: Continue providing great service."
    
    # Display result
    st.markdown("---")
    st.markdown(f"""
        <div class="{risk_class}">
            <h2>{emoji} {risk_level}</h2>
            <h1 style="font-size: 3rem; margin: 1rem 0;">{churn_prob:.1f}%</h1>
            <p style="font-size: 1.2rem;">{message}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Show details
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Will Churn", f"{churn_prob:.1f}%", delta=f"{churn_prob-50:.1f}%")
    
    with col2:
        st.metric("Will Stay", f"{no_churn_prob:.1f}%", delta=f"{no_churn_prob-50:.1f}%")
    
    # Show recommendation
    st.markdown(f"""
        <div class="info-box">
            <h4>Recommendation</h4>
            <p>{action}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Show what to do based on risk
    if churn_prob >= 70:
        st.error("⚠️ **Immediate Actions Needed:**")
        st.markdown("""
        - 🎁 Offer 15-20% discount on next purchase
        - 📞 Call customer to understand concerns
        - 🆓 Provide free shipping or exclusive perks
        - 💬 Send personalized email within 24 hours
        """)
    elif churn_prob >= 40:
        st.warning("⚠️ **Recommended Actions:**")
        st.markdown("""
        - 📧 Send targeted promotional email
        - 🎯 Show personalized product recommendations
        - 🌟 Invite to loyalty program
        - 📊 Monitor activity over next 2 weeks
        """)
    else:
        st.success("✅ **Maintenance Actions:**")
        st.markdown("""
        - ✉️ Continue regular newsletters
        - 🎉 Thank customer for their loyalty
        - 🌟 Request feedback or review
        - 🤝 Keep engaging on social media
        """)
    
   
