# Heart Disease Prediction System

## AI-ML Assignment 10 - End-to-End Machine Learning Model Deployment

### Objective
Develop a machine learning model that predicts whether a patient is at risk of heart disease based on clinical parameters, create a REST API using Flask, and deploy it as a live web service using Render.

### Dataset
- **Source:** [Heart Disease Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Records:** 303
- **Features:** 13 (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
- **Target:** target (0 = No Heart Disease, 1 = Heart Disease)

### Features Description
| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | Gender (1 = Male, 0 = Female) |
| cp | Chest Pain Type (0-3) |
| trestbps | Resting Blood Pressure (mm Hg) |
| chol | Serum Cholesterol (mg/dl) |
| fbs | Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No) |
| restecg | Resting ECG Results (0-2) |
| thalach | Maximum Heart Rate Achieved |
| exang | Exercise Induced Angina (1 = Yes, 0 = No) |
| oldpeak | ST Depression Induced by Exercise |
| slope | Slope of Peak Exercise ST Segment (0-2) |
| ca | Number of Major Vessels Colored by Fluoroscopy (0-3) |
| thal | Thalassemia (0-2) |

### Technologies Used
- **Python 3.x**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Scikit-learn** - Machine learning (Random Forest Classifier)
- **Joblib** - Model serialization
- **Flask** - REST API development
- **Render** - Cloud deployment

### Project Structure
```
HeartDiseaseDeployment/
│
├── app.py                 # Flask REST API
├── model.pkl              # Trained Random Forest model
├── scaler.pkl             # Feature scaler
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── train_model.py         # Model training script
├── heart.csv              # Dataset
├── templates/
│   └── index.html         # Web interface
└── static/                # Static files (optional)
```

### Installation & Setup

#### 1. Clone the repository
```bash
git clone https://github.com/yourusername/HeartDiseaseDeployment.git
cd HeartDiseaseDeployment
```

#### 2. Install dependencies
```bash
pip install -r requirements.txt
```

#### 3. Train the model
```bash
python train_model.py
```

#### 4. Run the Flask app locally
```bash
python app.py
```

### API Endpoints

#### 1. Home Page
- **URL:** `/`
- **Method:** GET
- **Description:** Web interface for predictions

#### 2. Make Prediction
- **URL:** `/predict`
- **Method:** POST
- **Content-Type:** application/json
- **Request Body:**
```json
{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
}
```
- **Response:**
```json
{
    "prediction": "Heart Disease Detected",
    "probability": 0.85
}
```

#### 3. Health Check
- **URL:** `/health`
- **Method:** GET
- **Response:**
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

### Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | 98.54% |
| Algorithm | Random Forest Classifier |
| n_estimators | 100 |

### Live Application
🔗 **Deployed URL:** [https://heartdiseasedeployment-y3j5.onrender.com](https://heartdiseasedeployment-y3j5.onrender.com)

### Conclusion

This project successfully developed a Random Forest classification model to predict heart disease risk based on clinical parameters, achieving approximately 85-90% accuracy. The model was deployed as a REST API using Flask and Render, making it accessible for real-time predictions.

Key challenges faced during deployment included ensuring consistent data preprocessing between training and inference, handling model file persistence on cloud platforms, and configuring the Flask application for production use with Gunicorn. These challenges highlighted the importance of proper MLOps practices.

MLOps is crucial for machine learning projects as it bridges the gap between model development and production deployment. It ensures reproducibility, enables continuous monitoring, and facilitates seamless updates to deployed models. This assignment demonstrated the complete ML lifecycle from data understanding to cloud deployment.

### References
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Render Documentation](https://render.com/docs)
- [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

### License
This project is for educational purposes as part of AI-ML Assignment 10.
