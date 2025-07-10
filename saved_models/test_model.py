
# Test script to verify saved models
import joblib
import pandas as pd
import json

# Load model and preprocessor
model = joblib.load('saved_models/best_model_lasso.pkl')

# Load feature info
with open('saved_models/feature_info.json', 'r') as f:
    feature_info = json.load(f)

# Create test data
test_data = pd.DataFrame({
    'gender': ['male'],
    'race_ethnicity': ['group B'],
    'parental_level_of_education': ["bachelor's degree"],
    'lunch': ['standard'],
    'test_preparation_course': ['none'],
    'math_score': [72],
    'reading_score': [72]
})

print("Test data:")
print(test_data)

try:
    # Make prediction
    prediction = model.predict(test_data)
    print(f"\nPrediction successful! Predicted score: {prediction[0]:.2f}")
except Exception as e:
    print(f"\nError during prediction: {e}")
    print("Model might need preprocessing or different input format")
