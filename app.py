from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
import traceback

app = FastAPI(title="Academic Success Predictor")

# Load models
try:
    model = joblib.load('models/tuned_models/Student_Model_v1.joblib')
    preprocessor = joblib.load('models/transformers/preprocessor.joblib')
    le = joblib.load('models/transformers/label_encoder.joblib')
except Exception as e:
    print(f"Error loading models: {e}")

class StudentData(BaseModel):
    Marital_status: int
    Application_mode: int
    Application_order: int
    Course: int
    Daytime_evening_attendance: int
    Previous_qualification: int
    Nacionality: int
    Mothers_qualification: int
    Fathers_qualification: int
    Mothers_occupation: int
    Fathers_occupation: int
    Displaced: int
    Educational_special_needs: int
    Debtor: int
    Tuition_fees_up_to_date: int
    Gender: int
    Scholarship_holder: int
    Age_at_enrollment: int
    International: int
    Curricular_units_1st_sem_credited: int
    Curricular_units_1st_sem_enrolled: int
    Curricular_units_1st_sem_evaluations: int
    Curricular_units_1st_sem_approved: int
    Curricular_units_1st_sem_grade: float
    Curricular_units_1st_sem_without_evaluations: int
    Curricular_units_2nd_sem_credited: int
    Curricular_units_2nd_sem_enrolled: int
    Curricular_units_2nd_sem_evaluations: int
    Curricular_units_2nd_sem_approved: int
    Curricular_units_2nd_sem_grade: float
    Curricular_units_2nd_sem_without_evaluations: int
    Unemployment_rate: float
    Inflation_rate: float
    GDP: float

@app.get("/")
def home():
    return {"message": "API is online."}

@app.post("/predict")
def predict(student: StudentData):
    try:
        # We must map our Pydantic fields to the EXACT column names from the CSV
        # This list must match the order and characters (/, ', ()) of your dataset.csv
        data_dict = {
            "Marital status": [student.Marital_status],
            "Application mode": [student.Application_mode],
            "Application order": [student.Application_order],
            "Course": [student.Course],
            "Daytime/evening attendance": [student.Daytime_evening_attendance],
            "Previous qualification": [student.Previous_qualification],
            "Nacionality": [student.Nacionality],
            "Mother's qualification": [student.Mothers_qualification],
            "Father's qualification": [student.Fathers_qualification],
            "Mother's occupation": [student.Mothers_occupation],
            "Father's occupation": [student.Fathers_occupation],
            "Displaced": [student.Displaced],
            "Educational special needs": [student.Educational_special_needs],
            "Debtor": [student.Debtor],
            "Tuition fees up to date": [student.Tuition_fees_up_to_date],
            "Gender": [student.Gender],
            "Scholarship holder": [student.Scholarship_holder],
            "Age at enrollment": [student.Age_at_enrollment],
            "International": [student.International],
            "Curricular units 1st sem (credited)": [student.Curricular_units_1st_sem_credited],
            "Curricular units 1st sem (enrolled)": [student.Curricular_units_1st_sem_enrolled],
            "Curricular units 1st sem (evaluations)": [student.Curricular_units_1st_sem_evaluations],
            "Curricular units 1st sem (approved)": [student.Curricular_units_1st_sem_approved],
            "Curricular units 1st sem (grade)": [student.Curricular_units_1st_sem_grade],
            "Curricular units 1st sem (without evaluations)": [student.Curricular_units_1st_sem_without_evaluations],
            "Curricular units 2nd sem (credited)": [student.Curricular_units_2nd_sem_credited],
            "Curricular units 2nd sem (enrolled)": [student.Curricular_units_2nd_sem_enrolled],
            "Curricular units 2nd sem (evaluations)": [student.Curricular_units_2nd_sem_evaluations],
            "Curricular units 2nd sem (approved)": [student.Curricular_units_2nd_sem_approved],
            "Curricular units 2nd sem (grade)": [student.Curricular_units_2nd_sem_grade],
            "Curricular units 2nd sem (without evaluations)": [student.Curricular_units_2nd_sem_without_evaluations],
            "Unemployment rate": [student.Unemployment_rate],
            "Inflation rate": [student.Inflation_rate],
            "GDP": [student.GDP]
        }

        df = pd.DataFrame(data_dict)
        
        # 1. Transform using the saved preprocessor
        X_processed = preprocessor.transform(df)
        
        # 2. Predict
        prediction_idx = model.predict(X_processed)
        
        # 3. Decode
        result = le.inverse_transform(prediction_idx)[0]
        
        return {"prediction": result}

    except Exception as e:
        # This will print the EXACT error in your VS Code terminal
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))