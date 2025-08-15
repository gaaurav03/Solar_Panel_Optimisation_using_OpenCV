# Solar Panel Tilt Angle Prediction Model

This project predicts the optimal tilt angle for solar panels based on weather and solar data. It includes machine learning models trained on historical solar radiation data.

## 📦 Model Files
The exported model package contains:
- `best_model.pkl` - The trained RandomForest/XGBoost model
- `feature_scaler.pkl` - Scaler for input features
- `lstm_model.h5` (optional) - The LSTM neural network model

## 🚀 How to Use

### 1. Installation
```bash
pip install -r requirements.txt

### 2. Loading the Model
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Example features (replace with your data)
sample_features = np.array([
    [23.5, 150, 0.75, 0.2, 850, 950, 0.85]  # Example row
])

# Preprocess and predict
scaled_features = scaler.transform(sample_features)
predicted_angle = model.predict(scaled_features)
print(f"Optimal tilt angle: {predicted_angle[0]:.2f}°")
