# app.py

import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Mobile Price Classification",
    page_icon="📱",
    layout="wide"
)

# =====================================================
# LOAD FILES
# =====================================================

rf_model = joblib.load(
    "models/random_forest_model.pkl"
)

selected_features = joblib.load(
    "models/selected_features.pkl"
)

comparison_df = pd.read_csv(
    "reports/model_comparison.csv"
)

# =====================================================
# PRICE LABELS
# =====================================================

price_labels = {
    0: "Low Cost",
    1: "Medium Cost",
    2: "High Cost",
    3: "Very High Cost"
}

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📱 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Dataset Insights",
        "Decision Tree Results",
        "Random Forest Results",
        "Feature Importance",
        "Predict Mobile Price"
    ]
)

# =====================================================
# HOME PAGE
# =====================================================

if page == "Home":

    st.title("📱 Mobile Price Range Classification")

    st.markdown("""
    Predict whether a smartphone belongs to:

    - Low Cost
    - Medium Cost
    - High Cost
    - Very High Cost

    using technical specifications and Machine Learning.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Dataset Rows",
            "2000"
        )

        st.metric(
            "Features Used",
            "14"
        )

    with col2:
        dt_acc = comparison_df.iloc[0]["Accuracy"]
        rf_acc = comparison_df.iloc[1]["Accuracy"]

        st.metric(
            "Decision Tree Accuracy",
            f"{dt_acc:.2%}"
        )

        st.metric(
            "Random Forest Accuracy",
            f"{rf_acc:.2%}"
        )

    st.success(
        "Random Forest was selected as the final model because it achieved the highest accuracy."
    )

# =====================================================
# DATASET PAGE
# =====================================================

elif page == "Dataset Insights":

    st.title("📊 Dataset Insights")

    st.subheader("Dataset Information")

    info_df = pd.DataFrame({
        "Property": [
            "Rows",
            "Features Used",
            "Target Classes"
        ],
        "Value": [
            2000,
            14,
            4
        ]
    })

    st.dataframe(info_df)

    st.subheader("Target Classes")

    st.markdown("""
    - **0 → Low Cost**
    - **1 → Medium Cost**
    - **2 → High Cost**
    - **3 → Very High Cost**
    """)

    st.subheader("Selected Features")

    feature_df = pd.DataFrame(
        selected_features,
        columns=["Feature"]
    )

    st.dataframe(feature_df)

# =====================================================
# DECISION TREE PAGE
# =====================================================

elif page == "Decision Tree Results":

    st.title("🌳 Decision Tree Results")

    dt_acc = comparison_df.iloc[0]["Accuracy"]

    st.metric(
        "Decision Tree Accuracy",
        f"{dt_acc:.2%}"
    )

    st.subheader("Classification Report")

    dt_report = pd.read_csv(
        "reports/classification_report_dt.csv"
    )

    st.dataframe(dt_report)

    st.subheader("Confusion Matrix")

    st.image(
        "images/confusion_matrix_dt.png"
    )

# =====================================================
# RANDOM FOREST PAGE
# =====================================================

elif page == "Random Forest Results":

    st.title("🌲 Random Forest Results")

    rf_acc = comparison_df.iloc[1]["Accuracy"]

    st.metric(
        "Random Forest Accuracy",
        f"{rf_acc:.2%}"
    )

    st.subheader("Classification Report")

    rf_report = pd.read_csv(
        "reports/classification_report_rf.csv"
    )

    st.dataframe(rf_report)

    st.subheader("Confusion Matrix")

    st.image(
        "images/confusion_matrix_rf.png"
    )

# =====================================================
# FEATURE IMPORTANCE PAGE
# =====================================================

elif page == "Feature Importance":

    st.title("📈 Feature Importance")

    st.markdown("""
    Random Forest determines which features contribute the most toward mobile price prediction.
    """)

    st.image(
        "images/feature_importance.png"
    )

    st.info(
        "RAM is typically the most influential feature for mobile price prediction."
    )

# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "Predict Mobile Price":

    st.title("📱 Predict Mobile Price Range")

    st.subheader("Basic Specifications")

    battery_power = st.number_input(
        "Battery Power (mAh)",
        value=5000,
        help="Battery capacity. Example: 5000 mAh"
    )

    ram_gb = st.selectbox(
    "RAM (GB)",
    [2, 3, 4, 6, 8, 12, 16, 24],
    index=4,
    help="Select the RAM capacity of the smartphone."
    )

    # Convert GB to MB because model was trained on MB

    ram = ram_gb * 1024

    int_memory = st.number_input(
        "Internal Storage (GB)",
        value=128,
        help="Examples: 64GB, 128GB, 256GB"
    )

    clock_speed = st.number_input(
        "Processor Speed (GHz)",
        value=2.5,
        step=0.1
    )

    n_cores = st.number_input(
        "CPU Cores",
        value=8
    )

    fc = st.number_input(
        "Front Camera (MP)",
        value=16
    )

    pc = st.number_input(
        "Rear Camera (MP)",
        value=64
    )

    st.subheader("Advanced Specifications")

    with st.expander("Open Advanced Settings"):

        px_height = st.number_input(
            "Screen Resolution Height (Pixels)",
            value=2400,
            help="Example: 2400"
        )

        px_width = st.number_input(
            "Screen Resolution Width (Pixels)",
            value=1080,
            help="Example: 1080"
        )

        sc_h = st.number_input(
            "Screen Height",
            value=18,
            help="Physical screen height feature used by dataset"
        )

        sc_w = st.number_input(
            "Screen Width",
            value=9,
            help="Physical screen width feature used by dataset"
        )

        mobile_wt = st.number_input(
            "Mobile Weight (grams)",
            value=180
        )

        m_dep = st.number_input(
            "Phone Thickness",
            value=0.5,
            step=0.1
        )

        talk_time = st.number_input(
            "Talk Time (Hours)",
            value=20,
            help="Estimated battery call duration"
        )

    if st.button("Predict Price Range"):

        input_df = pd.DataFrame([{
            "battery_power": battery_power,
            "clock_speed": clock_speed,
            "fc": fc,
            "int_memory": int_memory,
            "m_dep": m_dep,
            "mobile_wt": mobile_wt,
            "n_cores": n_cores,
            "pc": pc,
            "px_height": px_height,
            "px_width": px_width,
            "ram": ram,
            "sc_h": sc_h,
            "sc_w": sc_w,
            "talk_time": talk_time
        }])

        prediction = rf_model.predict(
            input_df
        )[0]

        probabilities = rf_model.predict_proba(
            input_df
        )[0]

        st.success(
            f"Predicted Category: {price_labels[prediction]}"
        )

        st.subheader("Prediction Confidence")

        for i, prob in enumerate(probabilities):

            st.write(
                f"{price_labels[i]} : {prob:.2%}"
            )

            st.progress(float(prob))