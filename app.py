import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# --------------------------
# Page Setup
# --------------------------
st.set_page_config(page_title="AI Diabetes Predictor Dashboard", layout="wide")

# --------------------------
# Load Data
# --------------------------
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

@st.cache_data
def load_model_results():
    return pd.read_csv("model_results.csv")

@st.cache_resource
def load_best_model():
    return joblib.load("best_model.pkl")

df = load_data()
results = load_model_results()
best_model = load_best_model()

# --------------------------
# Tabs
# --------------------------
tabs = st.tabs(["📊 Data Overview", "🤖 Model Comparison", "🔮 Live Prediction"])

# ==========================
# TAB 1 — DATA OVERVIEW
# ==========================
with tabs[0]:
    st.header("📊 Data Overview & Insights")
    st.write("This section provides an overview of the diabetes dataset and visual insights into key health parameters.")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Basic Statistics")
    st.write(df.describe())

    # ---- FIXED CORRELATION HEATMAP ----
    st.subheader("Correlation Heatmap")
    col_center = st.columns([1, 3, 1])  # centers chart
    with col_center[1]:
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
            square=True,
            ax=ax
        )
        ax.set_title("Feature Correlation Heatmap", fontsize=12, pad=10)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)

    # ---- IMPROVED TARGET VARIABLE DISTRIBUTION ----
    st.subheader("Distribution of Target Variable")
    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(x='Outcome', data=df, palette='coolwarm', ax=ax)
        ax.set_xticklabels(['Non-Diabetic', 'Diabetic'])
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.set_title("Target Variable Distribution", fontsize=12, pad=10)
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                        textcoords='offset points')
        st.pyplot(fig)

# ==========================
# TAB 2 — MODEL COMPARISON
# ==========================
with tabs[1]:
    st.header("🤖 Model Performance Comparison")
    st.write("This section compares accuracy and other metrics for all trained models.")

    # ---- Highlighted Metrics Table ----
    st.subheader("🏆 Model Metrics Table")
    styled_results = results.style.highlight_max(axis=0, color="lightgreen")
    st.dataframe(styled_results, use_container_width=True)

    # ---- Grouped Bar Chart ----
    st.subheader("📈 Model Metrics Comparison")

    melted = results.melt(id_vars=["Model"], var_name="Metric", value_name="Score")

    col_center = st.columns([1, 3, 1])
    with col_center[1]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x="Metric", y="Score", hue="Model", data=melted, palette="Blues_d", ax=ax)
        ax.set_title("Model Performance by Metric", fontsize=12, pad=10)
        ax.set_ylabel("Score")
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

    best_model_name = results.loc[results['Accuracy'].idxmax(), 'Model']
    st.success(f"🏆 **Best Performing Model:** {best_model_name}")

# ==========================
# TAB 3 — LIVE PREDICTION
# ==========================
with tabs[2]:
    st.header("🔮 Live Prediction")
    st.write("Enter health metrics below to predict if a person is likely to have diabetes.")

    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
        Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
        BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
        SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    with col2:
        Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        Age = st.number_input("Age", min_value=1, max_value=120, value=30)

    if st.button("Get Prediction"):
        user_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness,
                      Insulin, BMI, DiabetesPedigreeFunction, Age]]
        prediction = best_model.predict(user_data)[0]
        if prediction == 1:
            st.error("🔴 The person is **likely Diabetic**.")
        else:
            st.success("🟢 The person is **likely Non-Diabetic**.")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Machine Learning")
