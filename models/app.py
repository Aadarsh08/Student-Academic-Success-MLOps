from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Academic Success Predictor")

# Load models with correct paths based on your git status
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
        
        # 2. Map names back to EXACT CSV Column Names (Crucial for the preprocessor)
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
        
        # Rebuild dictionary with mapped keys
        final_input = {mapping.get(k, k): [v] for k, v in data.items()}
        
        # 3. Create DataFrame and Predict
        df = pd.DataFrame(final_input)
        processed_data = preprocessor.transform(df)
        prediction = model.predict(processed_data)
        
        # 4. Return human-readable label
        return {"prediction": le.inverse_transform(prediction)[0]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))