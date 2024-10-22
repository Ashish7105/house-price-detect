import joblib
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from sklearn.model_selection import train_test_split

# Example synthetic dataset for demonstration purposes
data = np.array([
    [1800, 4, 10, 500000],
    [2200, 5, 3, 750000],
    [1600, 3, 15, 400000],
    [3000, 6, 2, 900000],
    [1000, 2, 20, 300000],
    # Add more data as needed for better prediction
])

# Features: Size, Bedrooms, Age
X = data[:, :3]
# Target: Price
y = data[:, 3]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model (assuming you've already trained and saved it)
model = joblib.load('house_price_model.pkl')

def predict_price(size, bedrooms, age):
    # Prepare the input as a 2D array for the model
    input_data = [[size, bedrooms, age]]
    # Predict the price using the model
    predicted_price = model.predict(input_data)[0]
    return predicted_price

def plot_actual_vs_predicted(X_test, y_test):
    predictions = model.predict(X_test)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Price')
    plt.scatter(range(len(predictions)), predictions, color='red', label='Predicted Price')
    plt.xlabel('House Index')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Actual vs. Predicted House Prices')

    # Save the plot to a BytesIO object and convert to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url
