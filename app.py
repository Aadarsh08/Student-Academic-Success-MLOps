print("🚀 DEBUG: LOADED THE CORRECT NEW APP.PY")
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Academic Success Predictor")

# Load models - THESE PATHS MUST MATCH YOUR FOLDER
model = joblib.load('models/tuned_models/RandomForest_tuned.joblib')
preprocessor = joblib.load('models/transformers/preprocessor.joblib')
le = joblib.load('models/transformers/label_encoder.joblib')

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


@app.post("/predict")
def predict(student: StudentData):
    try:
        # 1. Convert Pydantic model to Dictionary
        data = student.model_dump()
        
        # 2. Complete Mapping (Covers every single column name mismatch)
        mapping = {
            "Marital_status": "Marital status",
            "Application_mode": "Application mode",
            "Application_order": "Application order",
            "Daytime_evening_attendance": "Daytime/evening attendance",
            "Previous_qualification": "Previous qualification",
            "Mothers_qualification": "Mother's qualification",
            "Fathers_qualification": "Father's qualification",
            "Mothers_occupation": "Mother's occupation",
            "Fathers_occupation": "Father's occupation",
            "Educational_special_needs": "Educational special needs",
            "Tuition_fees_up_to_date": "Tuition fees up to date",
            "Scholarship_holder": "Scholarship holder", # Fixed this missing line
            "Age_at_enrollment": "Age at enrollment",
            "Curricular_units_1st_sem_credited": "Curricular units 1st sem (credited)",
            "Curricular_units_1st_sem_enrolled": "Curricular units 1st sem (enrolled)",
            "Curricular_units_1st_sem_evaluations": "Curricular units 1st sem (evaluations)",
            "Curricular_units_1st_sem_approved": "Curricular units 1st sem (approved)",
            "Curricular_units_1st_sem_grade": "Curricular units 1st sem (grade)",
            "Curricular_units_1st_sem_without_evaluations": "Curricular units 1st sem (without evaluations)",
            "Curricular_units_2nd_sem_credited": "Curricular units 2nd sem (credited)",
            "Curricular_units_2nd_sem_enrolled": "Curricular units 2nd sem (enrolled)",
            "Curricular_units_2nd_sem_evaluations": "Curricular units 2nd sem (evaluations)",
            "Curricular_units_2nd_sem_approved": "Curricular units 2nd sem (approved)",
            "Curricular_units_2nd_sem_grade": "Curricular units 2nd sem (grade)",
            "Curricular_units_2nd_sem_without_evaluations": "Curricular units 2nd sem (without evaluations)",
            "Unemployment_rate": "Unemployment rate",
            "Inflation_rate": "Inflation rate"
        }
        
        # 3. Rebuild dictionary with EXACT column names from CSV
        final_input = {mapping.get(k, k): [v] for k, v in data.items()}
        
        # 4. Create DataFrame
        df = pd.DataFrame(final_input)
        
        # 5. Pipeline Preprocessing (Scaling/One-Hot)
        processed_data = preprocessor.transform(df)
        
        # 6. Model Prediction
        prediction = model.predict(processed_data)
        
        # 7. Decode and Return
        label = le.inverse_transform(prediction)[0]
        
        return {"prediction": label}
        
    except Exception as e:
        print(f"Error details: {e}")
        raise HTTPException(status_code=500, detail=str(e))