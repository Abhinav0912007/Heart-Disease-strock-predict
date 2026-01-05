import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

# ---------------- CSS (HOSPITAL THEME) ----------------
st.markdown("""
<style>
.main {
    background-color: #ffffff;
    padding: 20px;
}

/* Header */
.title {
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    color: #0d47a1;
}
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #555;
    margin-bottom: 25px;
}

/* Card */
.card {
    background: #f5faff;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.12);
}

/* Button */
.stButton>button {
    width: 100%;
    background-color: #0d47a1;
    color: white;
    font-size: 18px;
    padding: 12px;
    border-radius: 10px;
    border: none;
}
.stButton>button:hover {
    background-color: #1565c0;
}

/* Result Boxes */
.high-risk {
    background: #ffebee;
    color: #b71c1c;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: 700;
}
.low-risk {
    background: #e8f5e9;
    color: #1b5e20;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>❤️ Heart Disease Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered medical risk analysis | Developed by Abhinav</div>", unsafe_allow_html=True)

# ---------------- INPUT CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Heart Risk"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    # Risk percentage (simple interpretation)
    risk_percent = int((scaled_input.mean() + 1) * 50)
    risk_percent = max(5, min(risk_percent, 95))

    st.markdown("### 📊 Heart Disease Risk Level")
    st.progress(risk_percent / 100)
    st.markdown(f"**Estimated Risk: {risk_percent}%**")

    if prediction == 1:
        st.markdown("<div class='high-risk'>⚠️ High Risk Detected<br>Please consult a cardiologist.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='low-risk'>✅ Low Risk Detected<br>Maintain a healthy lifestyle.</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
    "<hr><p style='text-align:center; color:#777;'>⚕️ AI-based Medical Prediction System | © Abhinav Singh</p>",
    unsafe_allow_html=True
)
