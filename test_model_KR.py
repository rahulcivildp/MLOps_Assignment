from data_loader import load_data
import joblib

# Load dataset
df = load_data()

# Features only
X = df.drop("MEDV", axis=1)

# Keep as DataFrame
sample_data = X.iloc[:1]

# Load scaler
scaler = joblib.load(
    "saved_models/scaler.pkl"
)

# Scale data
sample_data = scaler.transform(sample_data)

# Load Kernel Ridge Regression model
model = joblib.load(
    "saved_models/Kernel Ridge Regressor.pkl"
)

# Predict
prediction = model.predict(sample_data)

print("Kernel Ridge Regression Prediction:")
print(prediction)