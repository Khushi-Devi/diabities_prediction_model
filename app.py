import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# --------------------------
# Page Setup
# --------------------------
st.set_page_config(
    page_title="AI Diabetes Predictor Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
            /* Main app background */
    .stApp {
    background: linear-gradient(to bottom, #0f172a 0%, #1e293b 100%);
    }

    /* Optional: Make content containers slightly transparent to show gradient */
    .block-container {
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 10px;
        padding: 2rem;
    }
    /* Main title styling */
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .main-header p {
        color: #f0f0f0;
        font-size: 1.1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a202c;
        color: #e2e8f0;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Input styling */
    .stNumberInput > div > div > input {
        border-radius: 8px;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem;
        font-size: 1.1rem;
    }
    
    /* Result boxes */
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        font-size: 1.2rem;
        animation: fadeIn 0.5s;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>🏥 AI Diabetes Predictor Dashboard</h1>
        <p>Empowering healthcare with Machine Learning insights</p>
    </div>
""", unsafe_allow_html=True)

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
    st.markdown("### 📊 Data Overview & Insights")
    st.markdown("Explore the diabetes dataset and visualize key health parameters")
    
    # Dataset info cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(df), border=True)
    with col2:
        st.metric("Features", len(df.columns) - 1, border=True)
    with col3:
        diabetic_count = df['Outcome'].sum()
        st.metric("Diabetic Cases", diabetic_count, border=True)
    with col4:
        non_diabetic = len(df) - diabetic_count
        st.metric("Non-Diabetic Cases", non_diabetic, border=True)
    
    st.markdown("---")
    
    # Dataset Preview with expander
    with st.expander("📋 Dataset Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    
    with st.expander("📈 Statistical Summary", expanded=False):
        st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("---")
    
    # Visualizations in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔥 Feature Correlation Heatmap")
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdYlBu_r",
            center=0,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            square=True,
            ax=ax
        )
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold', pad=15)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### 📊 Target Variable Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#667eea', '#f093fb']
        counts = df['Outcome'].value_counts()
        bars = ax.bar(['Non-Diabetic', 'Diabetic'], counts.values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({height/len(df)*100:.1f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel("Count", fontsize=12, fontweight='bold')
        ax.set_title("Distribution of Diabetes Outcome", fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)

# ==========================
# TAB 2 — MODEL COMPARISON
# ==========================
with tabs[1]:
    st.markdown("### 🤖 Model Performance Comparison")
    st.markdown("Compare accuracy and metrics across all trained machine learning models")
    
    # Best model highlight
    best_model_name = results.loc[results['Accuracy'].idxmax(), 'Model']
    best_accuracy = results['Accuracy'].max()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; 
                        color: white; margin-bottom: 2rem;'>
                <h2 style='margin: 0;'>🏆 Best Model</h2>
                <h3 style='margin: 0.5rem 0;'>{best_model_name}</h3>
                <p style='font-size: 1.5rem; margin: 0;'>Accuracy: {best_accuracy:.2%}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Metrics Table
    st.markdown("#### 📋 Detailed Metrics Table")
    styled_results = results.style.format({
        'Accuracy': '{:.4f}',
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1-Score': '{:.4f}'
    }).highlight_max(axis=0, color="lightgreen")
    st.dataframe(styled_results, use_container_width=True, height=250)
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(results['Model'], results['Accuracy'], color='#667eea', edgecolor='black')
        
        # Highlight best model
        max_idx = results['Accuracy'].idxmax()
        bars[max_idx].set_color('#f093fb')
        
        for i, (model, acc) in enumerate(zip(results['Model'], results['Accuracy'])):
            ax.text(acc + 0.005, i, f'{acc:.4f}', va='center', fontweight='bold')
        
        ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### 📈 All Metrics Comparison")
        melted = results.melt(id_vars=["Model"], var_name="Metric", value_name="Score")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x="Metric", y="Score", hue="Model", data=melted, 
                   palette="viridis", ax=ax, edgecolor='black')
        ax.set_ylabel("Score", fontsize=12, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=0)
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)

# ==========================
# TAB 3 — LIVE PREDICTION
# ==========================
with tabs[2]:
    st.markdown("### 🔮 Live Diabetes Prediction")
    st.markdown("Enter patient health metrics to predict diabetes likelihood using AI")
    
    st.markdown("---")
    
    # Input form with better organization
    st.markdown("#### 📝 Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**👤 Demographics**")
        Age = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=30, step=1
        )
        Pregnancies = st.number_input(
            "Pregnancies", min_value=0, max_value=20, value=2, step=1
        )
    
    with col2:
        st.markdown("**🩺 Vital Signs**")
        Glucose = st.number_input(
            "Glucose (mg/dL)", min_value=0, max_value=300, value=120, step=1
        )
        BloodPressure = st.number_input(
            "Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70, step=1
        )
        BMI = st.number_input(
            "BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1
        )
    
    with col3:
        st.markdown("**🔬 Lab Results**")
        Insulin = st.number_input(
            "Insulin (µU/mL)", min_value=0, max_value=900, value=80, step=5
        )
        SkinThickness = st.number_input(
            "Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1
        )
        DiabetesPedigreeFunction = st.number_input(
            "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01
        )
    
    st.markdown("---")
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("Analyzing patient data..."):
            user_data = [[
                Pregnancies, Glucose, BloodPressure, SkinThickness,
                Insulin, BMI, DiabetesPedigreeFunction, Age
            ]]
            prediction = best_model.predict(user_data)[0]
            probability = best_model.predict_proba(user_data)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if prediction == 1:
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); 
                                    padding: 2rem; border-radius: 15px; text-align: center; 
                                    color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                            <h2 style='margin: 0;'>⚠️ High Risk</h2>
                            <h3 style='margin: 1rem 0;'>Patient likely has Diabetes</h3>
                            <div style='background: rgba(255,255,255,0.2); padding: 1rem; 
                                        border-radius: 10px; margin-top: 1rem;'>
                                <p style='font-size: 1.1rem; margin: 0;'>Model Confidence</p>
                                <p style='font-size: 2rem; font-weight: bold; margin: 0.5rem 0;'>
                                    {probability[1]:.1%}
                                </p>
                            </div>
                            <p style='margin-top: 1rem; font-size: 0.9rem;'>
                                ⚕️ Recommendation: Consult a healthcare professional
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #51cf66 0%, #38b000 100%); 
                                    padding: 2rem; border-radius: 15px; text-align: center; 
                                    color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                            <h2 style='margin: 0;'>✅ Low Risk</h2>
                            <h3 style='margin: 1rem 0;'>Patient likely does not have Diabetes</h3>
                            <div style='background: rgba(255,255,255,0.2); padding: 1rem; 
                                        border-radius: 10px; margin-top: 1rem;'>
                                <p style='font-size: 1.1rem; margin: 0;'>Model Confidence</p>
                                <p style='font-size: 2rem; font-weight: bold; margin: 0.5rem 0;'>
                                    {probability[0]:.1%}
                                </p>
                            </div>
                            <p style='margin-top: 1rem; font-size: 0.9rem;'>
                                💚 Recommendation: Maintain healthy lifestyle
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Probability breakdown
                st.markdown("---")
                st.markdown("#### 📊 Detailed Probability Breakdown")
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    st.metric("Non-Diabetic Probability", f"{probability[0]:.2%}", border=True)
                with prob_col2:
                    st.metric("Diabetic Probability", f"{probability[1]:.2%}", border=True)

# ==========================
# Footer
# ==========================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #666;'>
        <p style='font-size: 0.9rem;'>
            Developed with ❤️ using <strong>Streamlit</strong> and <strong>Machine Learning</strong>
        </p>
        <p style='font-size: 0.8rem; margin-top: 0.5rem;'>
            ⚠️ <em>This tool is for educational purposes only. Always consult healthcare professionals for medical advice.</em>
        </p>
    </div>
""", unsafe_allow_html=True)