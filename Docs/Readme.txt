Solar Panel Tilt Angle Prediction Model:


This project predicts the optimal tilt angle for solar panels based on weather and solar data. It includes machine learning models trained on historical solar radiation data.

📦 Model Files
The exported model package contains:

best_model.pkl - The trained RandomForest/XGBoost model

feature_scaler.pkl - Scaler for input features

lstm_model.h5 (optional) - The LSTM neural network model

🚀 How to Use
1. Installation
bash
pip install -r requirements.txt
Requirements:

Python 3.8+

scikit-learn

pandas

numpy

xgboost

tensorflow (for LSTM model)

2. Loading the Model
python
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
3. Input Features
The model expects these 7 features in order:

Temperature (°C)

Day of year (1-365)

Clear sky radiation (W/m²)

Surface albedo

All-sky radiation (W/m²)

Clear-sky surface radiation (W/m²)

All-sky transparency index

🔧 How to Retrain
1. Prepare Data
python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('solar_data.csv')

# Clean data
df = df.replace(-999, np.nan).dropna()

# Features and target
X = df[['TEMP', 'DY', 'CLRSKY_SFC_SW_DWN', 'ALLSKY_SRF_ALB', 
        'ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 'ALLSKY_KT']]
y = df['calculated_tilt_angle']  # Your target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
2. Train Model
python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Scale features
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
predictions = model.predict(scaler.transform(X_test))
print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}°")
3. Hyperparameter Tuning (Optional)
python
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(RandomForestRegressor(), params, cv=5)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
📊 LSTM Model (Time Series)
For the LSTM model:

python
from tensorflow.keras.models import load_model

lstm_model = load_model('lstm_model.h5')

# Requires 3D input (samples, timesteps, features)
# Shape: (n, 120, 7) for 120 timesteps
predictions = lstm_model.predict(your_3d_input_array)
📝 Notes
All angle predictions are in degrees

Models expect cleaned data (no -999/missing values)

For time series predictions, maintain the 120-timestep sequence