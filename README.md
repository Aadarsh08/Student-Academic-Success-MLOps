# End-to-End MLOps Pipeline: Student Success Predictor

This project demonstrates a complete MLOps workflow — from data preprocessing and model training to deployment and monitoring — using a real-world dataset on student academic outcomes.

---

## 📌 Project Highlights
- **Problem Statement:** Predict student dropout, enrollment, or graduation risk.  
- **Dataset:** 76,518 rows, 38 features, imbalanced target (Graduate, Enrolled, Dropout).  
- **Tech Stack:** Python, Scikit-learn, MLflow, DVC, FastAPI, Docker, Hugging Face Spaces.  
- **MLOps Practices:** CI/CD with CML, model versioning, reproducibility, containerization.  

---

## 🔄 Pipeline Steps

### 1. Data Preprocessing
- Feature selection, encoding, imbalance handling.  
- Example:
```python      ****
X = df.drop(columns=[TARGET, 'id'])
y = df[TARGET]
```
## 2. Model Training & Hyperparameter Tuning
     a) Used GridSearchCV / RandomizedSearchCV for optimization.
     b) MLflow tracking for experiments.

## 3. Model Evaluation
     a) Metrics: Accuracy, F1-score, Confusion Matrix.
     b) Visualizations for performance analysis.

## 5. Experiment Tracking
     a) MLflow for model registry and versioning.

## 6. Continuous Integration
     a) GitHub Actions + CML for automated model reports.

## 7. Deployment
     a) FastAPI REST endpoint for serving predictions.
     b) Dockerized for portability.
     c) Hugging Face Spaces for live demo.

## 7. Monitoring
     a) Logs + performance tracking for deployed models.

## 📊 Results
    a) Balanced performance achieved after handling dataset imbalance.
    b) Deployed interactive predictor on Hugging Face Spaces:
    c) Live Demo - https://huggingface.co/spaces/Aadarsh-Mishra/student-success-predictor?utm_source=copilot.com

## ⚙️ How to Run Locally
```
git clone https://github.com/Aadarsh-Mishra/student-success-predictor
cd student-success-predictor
pip install -r requirements.txt
uvicorn app:app --reload
```

## 🚀 Future Improvements
    a) Integrate drift detection.
    b) Add automated retraining pipeline.
    c) Deploy on cloud (AWS/GCP/Azure).

## 📂 Repository Structure
Code
```student-success-predictor/
│── data/                # Raw and processed datasets
│── notebooks/           # Jupyter notebooks for exploration
│── src/                 # Source code (preprocessing, training, evaluation)
│── app/                 # FastAPI application
│── models/              # Saved models and MLflow registry
│── docker/              # Dockerfiles and configs
│── .github/workflows/   # CI/CD pipelines
│── requirements.txt     # Dependencies
│── README.md            # Project documentation
```
##🏷️ License
      This project is licensed under the MIT License — feel free to use and adapt.
