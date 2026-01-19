import streamlit as st
import pandas as pd
import joblib
import datetime
import plotly.graph_objects as go

MODEL_WITH_TESTS_PATH = 'PCOS_Modeltest.pkl'
MODEL_NO_TESTS_PATH = 'PCOS_Model_NoTests.pkl'

st.title("PCOS Prediction & Cramps Monitoring System")

# Model selection dropdown
model_choice = st.selectbox(
    "Choose prediction type:",
    ["With Tests", "Without Tests"]
)

# Load selected model
if model_choice == "With Tests":
    model_data = joblib.load(MODEL_WITH_TESTS_PATH)
else:
    model_data = joblib.load(MODEL_NO_TESTS_PATH)

model = model_data[0]
features = model_data[1]

default_values = {
    "Age": 23,
    "BMI": 34,
    "Cycle(R/I)": 0,              # 0=Regular
    "Cycle length (days)": 4,
    "Testosterone": 70.0,
    "LH": 15.0,
    "FSH": 6.0,
    "Pain(Y/N)": 0,
}

# Input form
st.subheader("Enter the required details")
user_data = {}
for feature in features:
    default = default_values.get(feature, 0)

    if "Y/N" in feature:
        user_data[feature] = st.selectbox(feature, [0, 1], index=int(default))
    elif "Cycle(R/I)" in feature:
        user_data[feature] = st.selectbox(feature, [0, 1], index=int(default))  # 0=Regular, 1=Irregular
    elif "days" in feature.lower():
        user_data[feature] = st.number_input(feature, min_value=0.0, value=float(default))
    else:
        user_data[feature] = st.number_input(feature, value=float(default))

# Prediction
if st.button("Predict PCOS"):
    input_df = pd.DataFrame([user_data], columns=features)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    if prediction == 1:
        st.error(f"High likelihood of PCOS ({probability:.2f}%) — Please consult a doctor.")
    else:
        st.success(f"Low likelihood of PCOS ({probability:.2f}%).")

# --------------------------------
# Cramps & Pain Monitoring Section
# --------------------------------
st.subheader("📊 Period Cramps & Pain Monitoring")

# Initialize session state
if "pain_data" not in st.session_state:
    st.session_state.pain_data = pd.DataFrame(columns=["Date", "Pain Level"])

# Input today's pain level
pain_level = st.slider("Pain Level (0 = No Pain, 10 = Severe)", 0, 10, 0)
if st.button("Log Pain Level"):
    today = datetime.date.today()
    new_row = pd.DataFrame([[today, pain_level]], columns=["Date", "Pain Level"])
    st.session_state.pain_data = pd.concat([st.session_state.pain_data, new_row], ignore_index=True)

# Show chart if data exists
if not st.session_state.pain_data.empty:
    df = st.session_state.pain_data.copy()

    # Latest pain level for gauge
    latest_pain = df["Pain Level"].iloc[-1]

    # Plot gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_pain,
        title={'text': "Today's Pain Level"},
        gauge={
            'axis': {'range': [0, 10]},
            'bar': {'color': "red" if latest_pain > 7 else "orange" if latest_pain > 3 else "green"},
            'steps': [
                {'range': [0, 3], 'color': 'green'},
                {'range': [3, 7], 'color': 'orange'},
                {'range': [7, 10], 'color': 'red'}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Optional: predict next day pain (simple rolling average)
    df["Pain Level Pred"] = df["Pain Level"].rolling(3, min_periods=1).mean()
    st.markdown(f"**Predicted Pain Level for Tomorrow:** {round(df['Pain Level Pred'].iloc[-1],1)} / 10")
