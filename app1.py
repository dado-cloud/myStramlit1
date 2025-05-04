import streamlit as st
import pandas as pd
import pickle

# Load trained model and scaler using pickle
model = pickle.load(open("mlp_model.pkl", "rb"))       
scaler = pickle.load(open("scaler.pkl", "rb"))      

# Application title
st.title("Student Depression Prediction App")
st.write("Enter the student's data to predict their likelihood of depression.")

# User inputs for features
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 10, 100, 18)
academic_pressure = st.slider("Academic Pressure", 0.0, 10.0, 5.0)
work_pressure = st.slider("Work Pressure", 0.0, 10.0, 5.0)
cgpa = st.slider("CGPA", 0.0, 5.0, 5.0, step=0.1)
study_satisfaction = st.slider("Study Satisfaction", 0.0, 5.0, 3.0)
job_satisfaction = st.slider("Job Satisfaction", 0.0, 5.0, 3.0)
sleep_duration = st.slider("Sleep Duration", 0, 12, 6)
dietary_habits = st.selectbox("Dietary Habits", ["Good", "Bad"])
degree = st.selectbox("Degree", ["Undergraduate", "Postgraduate"])
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
work_study_hours = st.slider("Work/Study Hours", 0, 24, 6)
financial_stress = st.slider("Financial Stress", 0.0, 10.0, 5.0)
family_history_mental_illness = st.selectbox("Family History of Mental Illness?", ["Yes", "No"])

# Prepare the input data into a DataFrame
input_df = pd.DataFrame({
    "Gender": [1 if gender == "Male" else 0],  # Convert 'Male' to 1 and 'Female' to 0
    "Age": [age],
    "AcademicPressure": [academic_pressure],
    "WorkPressure": [work_pressure],
    "CGPA": [cgpa],
    "StudySatisfaction": [study_satisfaction],
    "JobSatisfaction": [job_satisfaction],
    "SleepDuration": [sleep_duration],
    "DietaryHabits": [1 if dietary_habits == "Good" else 0],  # Convert 'Good' to 1 and 'Bad' to 0
    "Degree": [1 if degree == "Postgraduate" else 0],  # Convert 'Postgraduate' to 1 and 'Undergraduate' to 0
    "SuicidalThoughts": [1 if suicidal_thoughts == "Yes" else 0],  # Convert 'Yes' to 1 and 'No' to 0
    "WorkStudyHours": [work_study_hours],
    "FinancialStress": [financial_stress],
    "FamilyHistoryMentalIllness": [1 if family_history_mental_illness == "Yes" else 0]  # Convert 'Yes' to 1 and 'No' to 0
})

# Scale the input data using the previously trained scaler
scaled_input = scaler.transform(input_df)

# Prediction on button click
if st.button("Predict"):
    prediction = model.predict(scaled_input)[0]  # Predict the depression likelihood
    result = "Depressed" if prediction == 1 else "Not Depressed"
    st.subheader(f"Prediction: {result}")  # Display the result