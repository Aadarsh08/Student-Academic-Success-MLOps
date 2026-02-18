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
```python
X = df.drop(columns=[TARGET, 'id'])
y = df[TARGET]

2. Model Training & Hyperparameter Tuning
Used GridSearchCV / RandomizedSearchCV for optimization.

MLflow tracking for experiments.

3. Model Evaluation
Metrics: Accuracy, F1-score, Confusion Matrix.

Visualizations for performance analysis.

4. Experiment Tracking
MLflow for model registry and versioning.

5. Continuous Integration
GitHub Actions + CML for automated model reports.

6. Deployment
FastAPI REST endpoint for serving predictions.

Dockerized for portability.

Hugging Face Spaces for live demo.

7. Monitoring
Logs + performance tracking for deployed models.

📊 Results
Balanced performance achieved after handling dataset imbalance.

Deployed interactive predictor on Hugging Face Spaces:
👉 Live Demo - https://huggingface.co/spaces/Aadarsh-Mishra/student-success-predictor?utm_source=copilot.com

⚙️ How to Run Locally
git clone https://github.com/Aadarsh-Mishra/student-success-predictor
cd student-success-predictor
pip install -r requirements.txt
uvicorn app:app --reload

🚀 Future Improvements
Integrate drift detection.

Add automated retraining pipeline.

Deploy on cloud (AWS/GCP/Azure).

📂 Repository Structure
Code
student-success-predictor/
│── data/                # Raw and processed datasets
│── notebooks/           # Jupyter notebooks for exploration
│── src/                 # Source code (preprocessing, training, evaluation)
│── app/                 # FastAPI application
│── models/              # Saved models and MLflow registry
│── docker/              # Dockerfiles and configs
│── .github/workflows/   # CI/CD pipelines
│── requirements.txt     # Dependencies
│── README.md            # Project documentation

🏷️ License
This project is licensed under the MIT License — feel free to use and adapt.
