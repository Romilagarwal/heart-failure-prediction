# 🫀 Heart Failure Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.7+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://heart-vs.streamlit.app/)
[![Docker](https://img.shields.io/badge/Docker-Available-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-026F00?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

> **Live Demo**: [Streamlit Cloud App](https://heart-vs.streamlit.app/) | **Repository**: [GitHub](https://github.com/Romilagarwal/heart-failure-prediction)

## 📊 Project Overview

Cardiovascular diseases remain the **#1 cause of death globally**, claiming an estimated 17.9 million lives each year. Early detection and management are critical for people at high risk.

This machine learning system predicts heart disease risk based on clinical parameters, helping healthcare professionals identify at-risk patients earlier and potentially save lives through timely intervention.

<p align="center">
  <img src="https://github.com/Romilagarwal/heart-failure-prediction/blob/main/img/kenny-eliason-MEbT27ZrtdE-unsplash.jpg" width=300px alt="Heart Disease Visualization">
  <br>
  <i>Image: Kenny Eliason (Unsplash)</i>
</p>

## ✨ Key Features

- **High-accuracy prediction**: XGBoost classifier with fine-tuned parameters
- **Multiple deployment options**: Web application, API, and containerized solution
- **Interactive interface**: User-friendly Streamlit dashboard for instant predictions
- **RESTful API**: Flask backend for seamless integration with other systems
- **Docker support**: Ready for deployment in any environment
- **Cloud-deployed**: Accessible anywhere via Streamlit Cloud

## 📈 Dataset & Analysis

The model is trained on the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/) from Kaggle, containing key cardiovascular health indicators:

- Demographic details (age, sex)
- Clinical parameters (blood pressure, cholesterol levels)
- Symptoms (chest pain type, exercise-induced angina) 
- Test results (resting ECG, max heart rate, ST depression)

Our extensive [exploratory data analysis](https://github.com/Romilagarwal/heart-failure-prediction/blob/main/notebook/part_1_preprocessing.ipynb) reveals critical patterns between these variables and heart disease risk.

## 🧠 Model Development

The project evaluates multiple classification algorithms:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

After rigorous comparison using confusion matrices and AUC scoring, **XGBoost** emerged as the top performer. The [complete modeling process](https://github.com/Romilagarwal/heart-failure-prediction/blob/main/notebook/part_2_modeling.ipynb) includes feature importance analysis and hyperparameter tuning.

## 🚀 Deployment Options

### 1️⃣ Streamlit Web App
```bash
streamlit run heart_disease_prediction.py
```
Access a user-friendly interface to input patient data and receive instant predictions.

### 2️⃣ Flask API
```bash
python predict_flask.py
```
For programmatic access or integration with existing healthcare systems.

### 3️⃣ Docker Container
```bash
docker build -t heart-prediction-app .
docker run -it -p 9696:9696 --rm --name heart_app heart-prediction-app:latest
```
Deploy anywhere with consistent environment and dependencies.

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Quick Start
```bash
# Clone repository
git clone https://github.com/Romilagarwal/heart-failure-prediction.git
cd heart-failure-prediction

# Set up virtual environment
pip install pipenv
pipenv install
pipenv shell

# Run the app
streamlit run heart_disease_prediction.py
```

### Testing the API
```bash
python predict_flask_test.py
```

Example output:
```
{'hasHeartDisease': True, 'hasHeartDisease_probability': 0.73}
Potentially at risk of heart disease. Follow-up examination recommended.
```

## 📚 Additional Resources

- [Model training script](https://github.com/Romilagarwal/heart-failure-prediction/blob/main/training.py)
- [Docker installation guide](https://docs.docker.com/engine/install/)
- [Streamlit Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)

## ⚠️ Disclaimer

This tool is designed for **screening purposes only** and should not replace professional medical advice. Always consult healthcare providers for proper diagnosis.

## 🤝 Contributing

Contributions to improve the model accuracy, user interface, or add new features are welcome. Please feel free to submit a pull request.

## 📄 License

This project is available under the MIT License.

---

<p align="center">
  <i>Building healthier futures through machine learning and predictive analytics.</i>
</p>

