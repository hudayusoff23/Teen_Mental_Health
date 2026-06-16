import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="MindGuard AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM STYLING (Soft Blue-Green & Universal Legibility Fixes)
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Force clean font family globally */
html, body, [data-testid="stAppViewContainer"], .stApp, .stMarkdown, p, label, button {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hard override to prevent Streamlit from forcing dark mode backgrounds */
[data-testid="stAppViewContainer"] {
    background-color: #F4F7F6 !important;
    background-image: radial-gradient(#E2ECE9 1px, transparent 1px) !important;
    background-size: 20px 20px !important;
}

.main {
    padding-top: 1rem;
}

/* Sidebar Styling - Solid Deep Teal & White Text Override */
section[data-testid="stSidebar"] {
    background-color: #0E5E6F !important;
}

section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] h1,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] h3,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
    color: #FFFFFF !important;
}

/* Widget Container styling for visibility */
div[data-testid="stForm"] {
    background-color: #FFFFFF !important;
    border-radius: 12px;
    padding: 20px;
}

/* Fix main page input labels to be dark charcoal */
div[data-testid="stWidgetLabel"] p, 
label, 
.stSlider p, 
.stSelectbox p, 
.stNumberInput p,
.main div[data-testid="stMarkdownContainer"] p {
    color: #1F3A3A !important;
    font-weight: 500 !important;
}

/* CRITICAL FIX FOR ST.METRIC VISIBILITY */
div[data-testid="stMetricLabel"] p {
    color: #4A6B6B !important;  
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

div[data-testid="stMetricValue"] div {
    color: #0E5E6F !important;  
    font-weight: 700 !important;
    font-size: 2rem !important;
}

/* Specific styling for Section Titles */
h2 {
    color: #0E5E6F !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
    margin-top: 2rem !important;
    margin-bottom: 1rem !important;
    border-bottom: 2px solid #D1E5E4;
    padding-bottom: 5px;
}

h3 {
    color: #1F3A3A !important;
    font-weight: 600 !important;
    font-size: 1.2rem !important;
    margin-top: 1.5rem !important;
}

/* Action Button Layout & Text Fixes */
div.stButton > button:first-child {
    width: 100%;
    background: #00818A !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
    box-shadow: 0px 4px 10px rgba(0, 129, 138, 0.15) !important;
    transition: all 0.2s ease-in-out;
}

div.stButton > button:first-child *, 
div.stButton > button:first-child p,
div.stButton > button:first-child div {
    color: #FFFFFF !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

div.stButton > button:first-child:hover {
    background: #0E5E6F !important;
    transform: translateY(-1px);
    box-shadow: 0px 6px 15px rgba(0, 129, 138, 0.25) !important;
}

/* Force Expander Header text color to look clean */
details summary p {
    color: #0E5E6F !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_ml_components():
    return joblib.load("teen_mental_health_model.pkl")

components = load_ml_components()

model = components["model"]
le = components["encoder"]
feature_names = components["feature_names"]

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:

    st.markdown("# 🧠 MindGuard")

    st.markdown("""
    ### About

    MindGuard uses Machine Learning to evaluate
    teen behavioural, academic and social indicators
    that may be associated with mental health risk.

    ---
    
    ⚠️ Educational use only.
    
    This tool does not replace professional
    mental health assessment.
    """)

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------
st.markdown("""
<div style="
background: linear-gradient(135deg, #0E5E6F, #00818A);
padding: 35px 30px;
border-radius: 16px;
margin-bottom: 30px;
box-shadow: 0px 8px 24px rgba(14, 94, 111, 0.12);
">

<h1 style="color: #FFFFFF !important; margin: 0; font-family: 'Inter', sans-serif !important; font-weight: 700; font-size: 2.2rem;">
🧠 MindGuard AI
</h1>

<p style="font-size: 16px; color: #FFFFFF !important; margin-top: 8px; margin-bottom: 0; opacity: 0.95; font-family: 'Inter', sans-serif !important; font-weight: 400;">
Teen Mental Health Risk Assessment Dashboard
</p>

</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# STEP 1
# --------------------------------------------------
st.markdown("## 📋 Demographics & Lifestyle")

c1, c2, c3 = st.columns(3)

with c1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    age = st.slider(
        "Age",
        12, 19, 17
    )

with c2:
    platform = st.selectbox(
        "Primary Social Media Platform",
        ["Instagram","TikTok","YouTube","Threads","None"]
    )

    screen_time = st.slider(
        "Daily Screen Time (Hours)",
        0,15,4
    )

with c3:
    social_level = st.selectbox(
        "Physical Social Interaction",
        ["Low","Moderate","High"]
    )

    sleep_hours = st.slider(
        "Sleep Hours",
        3,12,5
    )

# --------------------------------------------------
# STEP 2
# --------------------------------------------------
st.markdown("## 📊 Psychological & Social Metrics")

c4, c5, c6 = st.columns(3)

with c4:
    academic_stress = st.slider(
        "Academic Stress",
        1,10,5
    )

    peer_pressure = st.slider(
        "Peer Pressure",
        1,10,5
    )

with c5:
    cyberbullying = st.slider(
        "Cyberbullying Experience",
        1,5,1
    )

    family_support = st.slider(
        "Family Support",
        1,5,4
    )

with c6:
    exercise_days = st.slider(
        "Exercise Days",
        0,7,2
    )

    online_friends = st.number_input(
        "Close Online Friends",
        min_value=0,
        max_value=500,
        value=10
    )

# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Analyze Mental Health"):

    gender_encoded = 0
    platform_encoded = 0
    social_encoded = 0

    input_data = pd.DataFrame([{
        'gender': gender_encoded,
        'platform_usage': platform_encoded,
        'social_interaction_level': social_encoded,
        'age': age,
        'sleep_hours': sleep_hours,
        'exercise_days': exercise_days,
        'online_friends': online_friends,
        'cyberbullying': cyberbullying,
        'family_support': family_support,
        'daily_social_media_hours': screen_time,
        'screen_time_before_sleep': 2,
        'academic_performance': 3,
        'physical_activity': exercise_days,
        'stress_level': academic_stress,
        'anxiety_level': peer_pressure,
        'addiction_level': 3
    }])

    input_data = input_data[feature_names]

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.markdown("---")
    st.markdown("## 🎯 Assessment Results")

    if prediction == 1:
        st.error(
            "⚠️ High Risk Profile Detected"
        )
    else:
        st.success(
            "✅ Low Risk Profile Detected"
        )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Analysis Certainty",
            f"{max(probabilities)*100:.1f}%"
        )

    with col2:
        st.metric(
            "Mental Distress Indicator",
            f"{probabilities[1]*100:.1f}%"
        )

    # FIXED CONTAINER: Injected an isolated light card background with explicit dark text mapping 
    with st.expander("ℹ️ What do these scores mean?"):
        st.markdown("""
        <div style="
            background-color: #FFFFFF !important; 
            border: 1px solid #D1E5E4 !important; 
            border-radius: 8px !important; 
            padding: 15px !important; 
            margin: 5px 0px !important;
        ">
            <p style="color: #1F3A3A !important; margin-bottom: 12px !important; font-size: 0.95rem; line-height: 1.5;">
                <strong style="color: #0E5E6F !important;">• Analysis Certainty (Consistency of Indicators):</strong> 
                This highlights how strongly and consistently the lifestyle factors, stress levels, and social indicators point 
                    toward a clear risk profile. A high certainty means the combination of factors (like low sleep paired with high 
                    cyberbullying) strongly reflects well-documented behavioral and emotional patterns.
            </p>
            <p style="color: #1F3A3A !important; margin-bottom: 0px !important; font-size: 0.95rem; line-height: 1.5;">
                <strong style="color: #0E5E6F !important;">• Mental Distress Indicator (Vulnerability Level):</strong> 
                This score estimates the overall emotional and psychological weight a teen might be carrying. Even if the 
                    main headline states a 'Low Risk Profile', a rising indicator percentage means that a combination of 
                    environmental stressors (such as low family support or high peer pressure) is beginning to build up, 
                    signaling a potential need for early supportive conversations before it escalates.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Risk Probability Trend")
    st.progress(float(probabilities[1]))

    st.info(
        "This assessment is generated by a machine learning model and should not be used as a medical diagnosis."
    )