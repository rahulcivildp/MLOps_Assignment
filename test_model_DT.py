from data_loader import load_data
import joblib

# Load dataset
df = load_data()

# Features only
X = df.drop("MEDV", axis=1)

# Convert to NumPy array
sample_data = X.iloc[:1].values

# Load saved model
model = joblib.load(
    "saved_models/Decision Tree Regressor.pkl"
)

# Predict
prediction = model.predict(sample_data)

print("Decision Tree Regressor Prediction:")
print(prediction)