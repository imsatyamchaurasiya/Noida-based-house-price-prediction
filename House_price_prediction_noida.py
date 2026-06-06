import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Noida House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp{
    background:#0f172a;
}

[data-testid="stSidebar"]{
    background:#111827;
}

[data-testid="stSidebar"] *{
    color:white !important;
}

[data-testid="stHeader"]{
    background:transparent;
}

h1,h2,h3,h4,h5,h6,p,label,span{
    color:white !important;
}

[data-testid="metric-container"]{
    background:#1e293b;
    border:1px solid #334155;
    padding:20px;
    border-radius:15px;
}

.stButton > button{
    width:100%;
    height:55px;
    background:#2563eb;
    color:white;
    border:none;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("nrpp.pkl", "rb"))

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.title("🏠 Noida Housing AI")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 Prediction",
            "📈 Analytics",
            "🤖 About"
        ]
    )

# ---------------- HOME PAGE ----------------
if page == "🏠 Home":

    st.markdown("""
    <h1 style='text-align:center'>
    🏠 Noida House Price Prediction
    </h1>

    <h3 style='text-align:center;color:#cbd5e1'>
    AI Powered Real Estate Analytics Platform
    </h3>
    """, unsafe_allow_html=True)

    st.write("")

    c1,c2,c3 = st.columns(3)

    c1.metric("Accuracy","94.36%")
    c2.metric("Algorithm","Random Forest")
    c3.metric("Records","500+")

    st.markdown("---")

    st.info("""
    This project predicts house prices based on:
    Sector, BHK, Area, Bathrooms, Floor, Furnishing and Metro Distance.
    """)

# ---------------- PREDICTION PAGE ----------------
elif page == "📊 Prediction":

    st.title("🏠 Property Details")

    col1,col2 = st.columns(2)

    with col1:

        sector = st.number_input(
            "Sector",
            min_value=1,
            max_value=150,
            value=62
        )

        bhk = st.selectbox(
            "BHK",
            [1,2,3,4,5]
        )

        area = st.number_input(
            "Area (sq ft)",
            min_value=500,
            max_value=5000,
            value=1000
        )

        bathrooms = st.selectbox(
            "Bathrooms",
            [1,2,3,4,5]
        )

    with col2:

        floor = st.number_input(
            "Floor",
            min_value=0,
            max_value=40,
            value=5
        )

        furnishing = st.selectbox(
            "Furnishing",
            [
                "Unfurnished",
                "Semi Furnished",
                "Fully Furnished"
            ]
        )

        metro = st.slider(
            "Metro Distance (km)",
            0.0,
            20.0,
            1.0
        )

    furnish_map = {
        "Unfurnished":0,
        "Semi Furnished":1,
        "Fully Furnished":2
    }

    if st.button("🚀 Predict House Price"):

        input_data = np.array([[
            sector,
            bhk,
            area,
            bathrooms,
            floor,
            furnish_map[furnishing],
            metro
        ]])

        prediction = model.predict(input_data)[0]

        st.success(
            f"🏠 Estimated House Price : ₹ {prediction:.2f} Lakhs"
        )

        report = pd.DataFrame({
            "Sector":[sector],
            "BHK":[bhk],
            "Area":[area],
            "Bathrooms":[bathrooms],
            "Floor":[floor],
            "Furnishing":[furnishing],
            "Metro Distance":[metro],
            "Predicted Price":[prediction]
        })

        st.download_button(
            "📥 Download Report",
            report.to_csv(index=False),
            "prediction_report.csv",
            "text/csv"
        )

# ---------------- ANALYTICS PAGE ----------------
elif page == "📈 Analytics":

    st.title("📈 Analytics Dashboard")

    c1,c2,c3 = st.columns(3)

    c1.metric("Accuracy","94.36%")
    c2.metric("Model","Random Forest")
    c3.metric("Dataset","500+")

# ---------------- ABOUT PAGE ----------------
else:

    st.title("🤖 About Project")

    st.write("""
    ### Noida House Price Prediction

    End-to-End Machine Learning Project

    #### Technology Stack
    - Python
    - Pandas
    - NumPy
    - Streamlit
    - Scikit-Learn
    - Pickle

    #### Algorithm
    - Random Forest Regressor

    #### Performance
    - R² Score: 94.36%
    - MAE: 7.86

    #### Developer
    Satyam Chaurasiya
    """)

st.markdown("---")
st.markdown(
    "<center><h4 style='color:white'>🚀 Developed by Satyam Chaurasiya</h4></center>",
    unsafe_allow_html=True
)