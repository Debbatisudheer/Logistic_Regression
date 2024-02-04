import pandas as pd
import joblib

# Load the trained model and scaler
loaded_model = joblib.load('main.pkl')
scaler = joblib.load('scaler.pkl')

# Create a DataFrame with the new person's information
new_person_data = pd.DataFrame({
    'Pregnancies':[1],
    'Glucose': [85],
    'BloodPressure': [66],
    'SkinThickness': [29],
    'Insulin': [0],
    'BMI': [22.6],
    'DiabetesPedigreeFunction': [0.351],
    'Age': [31],
})

# Scale the new data using the loaded scaler
X_new_person_scaled = scaler.transform(new_person_data)

# Make predictions on the new data
prediction = loaded_model.predict(X_new_person_scaled)

# Display the prediction
print("Prediction for the New Person:", prediction[0])