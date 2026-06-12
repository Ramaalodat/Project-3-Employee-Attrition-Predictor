import streamlit as st
import pandas as pd
import pickle
import os

# ──────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Predictor | DecodeLabs",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
#  CUSTOM CSS (التصميم الذي اتفقنا عليه)
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #1a103c, #120e2e); color: #f0f0f0; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); border-right: 1px solid #0f3460; }
    .hero-banner { background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 50%, #533483 100%); border: 1px solid #e94560; border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 2rem; text-align: center; box-shadow: 0 8px 32px rgba(233, 69, 96, 0.2); }
    .hero-banner h1 { font-size: 2.5rem; font-weight: 800; background: linear-gradient(90deg, #e94560, #a78bfa, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
    .hero-banner p { color: #a0aec0; font-size: 1rem; margin-top: 0.5rem; }
    .section-title { font-size: 1.1rem; font-weight: 700; color: #a78bfa; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 1rem; border-bottom: 1px solid rgba(167,139,250,0.3); padding-bottom: 0.5rem; }
    .result-card { background: rgba(26, 26, 58, 0.65); border: 1px solid rgba(167, 139, 250, 0.25); border-radius: 16px; padding: 1.5rem 2rem; margin-top: 1.2rem; transition: all 0.3s ease; box-shadow: 0 4px 25px rgba(0,0,0,0.4); text-align: center; }
    .result-card h2 { margin: 0; font-size: 2rem; }
    .stButton > button { background: linear-gradient(135deg, #e94560, #a78bfa); color: white; border: none; border-radius: 10px; padding: 0.6rem 2rem; font-weight: 700; font-size: 1rem; width: 100%; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(233,69,96,0.3); margin-top: 1.5rem; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(233,69,96,0.5); }
    div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label { color: #cbd5e0 !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  LOAD MODEL
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "attrition_model.pkl")
    with open(model_path, 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error(" Model file not found! Please run train_model.py first.")
    st.stop()

# ──────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏢 DecodeLabs HR")
    st.markdown("**AI Project 3** — Employee Classification")
    st.markdown("---")
    st.markdown("#### ⚙️ Classification Model")
    st.info("Algorithm: **Random Forest Classifier**\n\nThis model is trained to classify an employee into two categories:\n1. **Attrited** (Will Leave)\n2. **Retained** (Will Stay)")
    st.markdown("---")
    st.markdown("<p style='color:#718096; font-size:0.8rem; text-align:center;'>Powered by DecodeLabs · Batch 2026</p>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  HERO BANNER
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🏢 Employee Attrition Predictor</h1>
    <p>Enter employee data → Let AI classify if they are at risk of leaving the company</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  INPUT SECTION
# ──────────────────────────────────────────────
st.markdown('<div class="section-title"> Employee Profile</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=65, value=30, help="Employee's age")
    years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=5)
    overtime = st.selectbox("Does OverTime?", ["No", "Yes"])

with col2:
    monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000)
    distance_from_home = st.number_input("Distance From Home (km)", min_value=1, max_value=50, value=10)

with col3:
    job_satisfaction = st.slider("Job Satisfaction", min_value=1, max_value=4, value=3, help="1=Low, 4=Very High")
    work_life_balance = st.slider("Work-Life Balance", min_value=1, max_value=4, value=3, help="1=Bad, 4=Excellent")

# ──────────────────────────────────────────────
#  PREDICTION LOGIC
# ──────────────────────────────────────────────
run_btn = st.button(" Predict Employee Status")

if run_btn:
    # 1. تجهيز البيانات بنفس الترتيب الذي تدرب عليه النموذج
    input_data = pd.DataFrame([[
        age, 
        monthly_income, 
        job_satisfaction, 
        years_at_company, 
        1 if overtime == "Yes" else 0, 
        distance_from_home, 
        work_life_balance
    ]], columns=['Age', 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany', 'OverTime', 'DistanceFromHome', 'WorkLifeBalance'])
    
    # 2. عمل التصنيف (Classification)
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # 3. عرض النتيجة بطريقة فخمة
    st.markdown("---")
    st.markdown('<div class="section-title">🎯 Classification Result</div>', unsafe_allow_html=True)
    
    if prediction == 1:
        # Attrition = Yes
        confidence = round(probabilities[1] * 100, 1)
        st.markdown(f"""
        <div class="result-card" style="border-color: #e94560;">
            <h2 style="color: #e94560;">⚠️ High Risk of Attrition</h2>
            <p style="color: #cbd5e0; font-size: 1.1rem; margin-top: 10px;">The AI classifies this employee as likely to <b>leave</b> the company.</p>
            <div style="color: #fca5a5; font-size: 0.9rem; margin-top: 15px;">Confidence Score: <b>{confidence}%</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Attrition = No
        confidence = round(probabilities[0] * 100, 1)
        st.markdown(f"""
        <div class="result-card" style="border-color: #60a5fa;">
            <h2 style="color: #60a5fa;">✅ Retained (Safe)</h2>
            <p style="color: #cbd5e0; font-size: 1.1rem; margin-top: 10px;">The AI classifies this employee as likely to <b>stay</b> with the company.</p>
            <div style="color: #93c5fd; font-size: 0.9rem; margin-top: 15px;">Confidence Score: <b>{confidence}%</b></div>
        </div>
        """, unsafe_allow_html=True)