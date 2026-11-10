
import streamlit as st
import pandas as pd
import joblib
import json

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

# Load metadata
@st.cache_data
def load_metadata():
    with open("metadata.json", "r") as file:
        return json.load(file)

model = load_model()
metadata = load_metadata()

# Application heading
st.title("🩺 Diabetes Prediction System")

st.write(
    "Enter the required health measurements to get "
    "a prediction from the trained machine-learning model."
)

st.info(
    "This application is an educational machine-learning "
    "demonstration. It does not diagnose diabetes."
)

st.divider()

st.subheader("Enter Patient Information")

defaults = metadata["default_values"]

# Input form
with st.form("diabetes_prediction_form"):

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        value=int(round(defaults["Pregnancies"])),
        step=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        value=float(defaults["Glucose"])
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        value=float(defaults["BloodPressure"])
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        value=float(defaults["SkinThickness"])
    )

    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        value=float(defaults["Insulin"])
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        value=float(defaults["BMI"])
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        value=float(defaults["DiabetesPedigreeFunction"]),
        format="%.4f"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        value=max(1, int(round(defaults["Age"]))),
        step=1
    )

    submitted = st.form_submit_button(
        "Predict Diabetes",
        use_container_width=True
    )

# Make prediction
if submitted:

    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=metadata["features"]
    )

    prediction = int(model.predict(input_data)[0])

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Model Prediction: Diabetic")
        st.write(
            "The model classified this input as class 1."
        )
    else:
        st.success("Model Prediction: Non-Diabetic")
        st.write(
            "The model classified this input as class 0."
        )

    st.caption(
        "This result is not a medical diagnosis. "
        "Consult a qualified healthcare professional "
        "for medical assessment."
    )

# Model information
with st.expander("Model Information"):
    st.write("Algorithm:", metadata["algorithm"])
    st.write("Kernel:", metadata["kernel"])
    st.write(
        "Training Accuracy:",
        f"{metadata['evaluation']['training_accuracy'] * 100:.2f}%"
    )
    st.write(
        "Testing Accuracy:",
        f"{metadata['evaluation']['testing_accuracy'] * 100:.2f}%"
    )
