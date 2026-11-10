
# Diabetes Prediction Using Machine Learning

A machine learning project that predicts diabetes classification
using a Support Vector Machine (SVM) classifier.

## Algorithm
- Support Vector Machine
- Linear Kernel
- StandardScaler for feature standardization

## Dataset
PIMA Diabetes Dataset

- Total records: 768
- Input features: 8
- Target column: Outcome

## Input Features
1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age

## Target Classes
- 0: Non-Diabetic
- 1: Diabetic

## Project Files
- app.py: Streamlit web application
- model.pkl: Trained SVM model and scaler pipeline
- requirements.txt: Required Python packages
- metadata.json: Model information and evaluation results

## Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

## Deployment

1. Upload the project files to GitHub.
2. Open https://share.streamlit.io/
3. Connect your GitHub account.
4. Select this repository.
5. Select app.py as the main file.
6. Click Deploy.

## Disclaimer
This project is for educational purposes only.
It is not a substitute for professional medical diagnosis.
