import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Create synthetic dataset
data = {
    'Size': [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
    'Bedrooms': [3, 3, 3, 4, 4, 4, 5, 5, 5, 6],
    'Age': [10, 15, 20, 10, 5, 2, 3, 8, 15, 12],
    'Price': [300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000, 460000, 480000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Prepare the data
X = df[['Size', 'Bedrooms', 'Age']]
y = df['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'house_price_model.pkl')

def predict_price(size, bedrooms, age):
    model = joblib.load('house_price_model.pkl')
    return model.predict(np.array([[size, bedrooms, age]]))[0]
