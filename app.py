
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg,
        #1e3c72,
        #2a5298,
        #6dd5ed
    );
}

h1, h2, h3 {
    color: white !important;
}

p, label {
    color: white !important;
}

[data-testid="stMetricValue"] {
    color: #00ff99;
}

[data-testid="stMetricLabel"] {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
model = joblib.load("house_price_model.pkl")

# ---------------------------------------------------
# LOAD DATASET (OPTIONAL)
# ---------------------------------------------------
df = None

if os.path.exists("housing.csv"):
    df = pd.read_csv("housing.csv")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("🏠 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Dataset Analysis",
        "House Price Prediction",
        "Model Performance"
    ]
)

# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------
if page == "Home":

    st.title("🏠 California House Price Prediction")

    st.markdown("""
    ### Machine Learning Project

    Predict California housing prices using an XGBoost Regressor.

    #### Features
    - House Price Prediction
    - Dataset Analysis
    - Interactive Dashboard
    - Feature Importance Visualization
    - Model Performance Metrics

    #### Technologies
    - Python
    - Scikit-Learn
    - XGBoost
    - Pandas
    - Streamlit
    - Plotly
    """)

# ---------------------------------------------------
# DATASET ANALYSIS
# ---------------------------------------------------
elif page == "Dataset Analysis":

    st.title("📊 Dataset Analysis")

    if df is not None:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Houses",
            f"{len(df):,}"
        )

        col2.metric(
            "Features",
            df.shape[1] - 1
        )

        col3.metric(
            "Average Price",
            f"${df.iloc[:, -1].mean()*100000:,.0f}"
        )

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("House Price Distribution")

        target_col = df.columns[-1]

        fig = px.histogram(
            df,
            x=target_col,
            nbins=50,
            title="Distribution of House Prices"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        if "MedInc" in df.columns:

            st.subheader(
                "Median Income vs House Price"
            )

            fig2 = px.scatter(
                df.sample(
                    min(2000, len(df))
                ),
                x="MedInc",
                y=target_col,
                opacity=0.6
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

    else:

        st.warning(
            "housing.csv not found. Dataset analysis unavailable."
        )

# ---------------------------------------------------
# HOUSE PRICE PREDICTION
# ---------------------------------------------------
elif page == "House Price Prediction":

    st.title("🏡 Predict House Price")

    col1, col2 = st.columns(2)

    with col1:
        MedInc = st.number_input(
            "💵 Median Income",
            value=3.5
        )

        HouseAge = st.number_input(
            "🏠 House Age",
            value=20.0
        )

        AveRooms = st.number_input(
            "🛋 Average Rooms",
            value=5.0
        )

        AveBedrms = st.number_input(
            "🛏 Average Bedrooms",
            value=1.0
        )

    with col2:
        Population = st.number_input(
            "👨‍👩‍👧 Population",
            value=1000.0
        )

        AveOccup = st.number_input(
            "🏘 Average Occupancy",
            value=3.0
        )

        Latitude = st.number_input(
            "📍 Latitude",
            value=34.0
        )

        Longitude = st.number_input(
            "🌎 Longitude",
            value=-118.0
        )

    if st.button("🔮 Predict Price"):

        features = np.array([
            [
                MedInc,
                HouseAge,
                AveRooms,
                AveBedrms,
                Population,
                AveOccup,
                Latitude,
                Longitude
            ]
        ])

        prediction = model.predict(features)

        st.success(
            f"🏠 Estimated House Price: ${prediction[0]*100000:,.2f}"
        )

# ---------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------
elif page == "Model Performance":

    st.title("📈 Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Training R²",
        "0.9437"
    )

    col2.metric(
        "Test R²",
        "0.8338"
    )

    col3.metric(
        "Test MAE",
        "$31,090"
    )

    st.success(
        "The XGBoost Regressor performs well on unseen data and explains over 83% of house-price variance."
    )

    if hasattr(model, "feature_importances_"):

        feature_names = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude"
        ]

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        )

        st.subheader("Feature Importance")

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
