from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('OneHotEncoded_Yield_df.csv')

# Train the linear regression model
regressor = LinearRegression()
regressor.fit(data.iloc[:, :-1], data.iloc[:, -1])

# Define the route for the home page
@app.route('/')
def home():
    return render_template('C:/Users/harsh/OneDrive/Desktop/MIni Project/home.html')

# Define the route for the prediction result page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user inputs from the form
    temperature = float(request.form['temperature'])
    rainfall = float(request.form['rainfall'])
    humidity = float(request.form['humidity'])
    crop_type = int(request.form['crop_type'])

    # Make a prediction using the trained model
    prediction = regressor.predict([[temperature, rainfall, humidity, crop_type]])

    # Render the prediction result template with the predicted yield
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
