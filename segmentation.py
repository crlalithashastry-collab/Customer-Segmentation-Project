# ============================================================
# CUSTOMER SEGMENTATION WEB APPLICATION
# K-Means Clustering + Streamlit
# ============================================================

import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }

    .result-title {
        font-size: 18px;
        font-weight: 600;
    }

    .cluster-number {
        font-size: 40px;
        font-weight: 700;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

@st.cache_resource
def load_models():

    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

    return model, scaler


try:

    kmeans, scaler = load_models()

except FileNotFoundError:

    st.error(
        "Model files were not found. "
        "Please run customer_segmentation.py first."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("👥 Customer Segmentation")

    st.markdown("---")

    st.subheader("About the Project")

    st.write(
        """
        This application uses **K-Means Clustering**
        to group customers based on their demographic
        and purchasing behavior.
        """
    )

    st.markdown("---")

    st.subheader("Features Used")

    st.write("• Age")
    st.write("• Income")
    st.write("• Total Spending")
    st.write("• Web Purchases")
    st.write("• Store Purchases")
    st.write("• Web Visits per Month")

    st.markdown("---")

    st.caption(
        "Machine Learning Project"
    )

    st.caption(
        "Python • Scikit-learn • Streamlit"
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">👥 Customer Segmentation</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Predict the customer segment using K-Means clustering
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INFORMATION CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Algorithm",
        "K-Means"
    )

with col2:

    st.metric(
        "Customer Segments",
        "6"
    )

with col3:

    st.metric(
        "Model Status",
        "Ready"
    )


st.markdown("---")


# ============================================================
# CUSTOMER INPUT SECTION
# ============================================================

st.header("📝 Enter Customer Information")

st.write(
    "Enter the customer's details below and click "
    "**Predict Customer Segment**."
)


# First row

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40,
        step=1
    )


with col2:

    income = st.number_input(
        "Income",
        min_value=0.0,
        max_value=200000.0,
        value=50000.0,
        step=1000.0
    )


with col3:

    total_spending = st.number_input(
        "Total Spending",
        min_value=0.0,
        max_value=10000.0,
        value=800.0,
        step=50.0
    )


# Second row

col1, col2, col3 = st.columns(3)


with col1:

    web_purchases = st.number_input(
        "Web Purchases",
        min_value=0,
        max_value=100,
        value=10,
        step=1
    )


with col2:

    store_purchases = st.number_input(
        "Store Purchases",
        min_value=0,
        max_value=100,
        value=8,
        step=1
    )


with col3:

    web_visits = st.number_input(
        "Web Visits per Month",
        min_value=0,
        max_value=100,
        value=6,
        step=1
    )


st.markdown("---")


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔍 Predict Customer Segment",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # Create dataframe using EXACTLY
    # the same features used during training.

    input_data = pd.DataFrame(
        {
            "Age": [age],

            "Income": [income],

            "Total_Spending": [total_spending],

            "NumWebPurchases": [web_purchases],

            "NumStorePurchases": [store_purchases],

            "NumWebVisitsMonth": [web_visits]
        }
    )


    # Scale the input

    input_scaled = scaler.transform(
        input_data
    )


    # Predict cluster

    prediction = kmeans.predict(
        input_scaled
    )


    cluster = int(
        prediction[0]
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.success(
        "Customer segment predicted successfully!"
    )

    st.markdown(
        f"""
        <div class="result-box">

        <div class="result-title">
        🎯 Predicted Customer Segment
        </div>

        <div class="cluster-number">
        Cluster {cluster}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # CUSTOMER SUMMARY
    # ========================================================

    st.subheader("📊 Customer Summary")


    summary_col1, summary_col2, summary_col3 = st.columns(3)


    with summary_col1:

        st.metric(
            "Age",
            f"{age}"
        )

        st.metric(
            "Income",
            f"{income:,.0f}"
        )


    with summary_col2:

        st.metric(
            "Total Spending",
            f"{total_spending:,.0f}"
        )

        st.metric(
            "Web Purchases",
            f"{web_purchases}"
        )


    with summary_col3:

        st.metric(
            "Store Purchases",
            f"{store_purchases}"
        )

        st.metric(
            "Web Visits",
            f"{web_visits}"
        )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.subheader("💡 Segment Information")

    st.info(
        f"""
        This customer has been assigned to **Cluster {cluster}**
        based on similarities in age, income, spending,
        online purchases, store purchases, and web visits.

        The cluster number is generated by the K-Means model.
        Use the `cluster_summary.csv` file to study the
        detailed characteristics of each cluster.
        """
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.markdown("---")

st.header("📌 Project Overview")

overview_col1, overview_col2 = st.columns(2)


with overview_col1:

    st.subheader("Objective")

    st.write(
        """
        Customer segmentation helps businesses identify
        groups of customers with similar characteristics
        and purchasing behavior.
        """
    )


with overview_col2:

    st.subheader("Business Applications")

    st.write(
        """
        • Targeted marketing

        • Personalized offers

        • Customer prioritization

        • Campaign planning

        • Customer behavior analysis
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Customer Segmentation using K-Means Clustering | "
    "Python + Scikit-learn + Streamlit"
)