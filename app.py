from flask import Flask, render_template, request
from model import predict_price

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    size = float(request.form['size'])
    bedrooms = int(request.form['bedrooms'])
    age = int(request.form['age'])
    
    predicted_price = predict_price(size, bedrooms, age)
    return render_template('result.html', price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)
