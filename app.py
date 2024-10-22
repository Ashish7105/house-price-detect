from flask import Flask, render_template, request
from model import predict_price, plot_actual_vs_predicted, X_test, y_test

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    size = float(request.form['size'])
    bedrooms = int(request.form['bedrooms'])
    age = int(request.form['age'])

    # Get the predicted price
    predicted_price = predict_price(size, bedrooms, age)
    # Get the plot URL for actual vs predicted prices
    plot_url = plot_actual_vs_predicted(X_test, y_test)

    return render_template('result.html', price=predicted_price, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
