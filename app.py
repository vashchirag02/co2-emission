from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('Co2_Emissions.joblib')

# Load the dataset to get dropdown options
data = pd.read_csv('analyzed_dataset.csv')
countries = data['Country'].unique()

@app.route('/')
def index():
    return render_template('index.html', countries=countries)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        country = request.form['country']
        year = int(request.form['year'])
        gdp = float(request.form['gdp'])
        energy_consumption = float(request.form['energy_consumption'])
        electricity_generation = float(request.form['electricity_generation'])

        # Prepare the input data
        input_data = pd.DataFrame({
            'GDP, PPP (constant 2017 international $)': [gdp],
            'Primary energy consumption (TWh)': [energy_consumption],
            'Electricity Generation (TWh)': [electricity_generation]
        })

        # Predict the CO2 emissions
        prediction = model.predict(input_data)[0]

        # Classification logic
        high_threshold = 1200
        low_threshold = 1000

        if prediction > high_threshold:
            classification = 'High'
        elif prediction < low_threshold:
            classification = 'Low'
        else:
            classification = 'Average'

        # Render the result
        return render_template('result.html', prediction=prediction, classification=classification, country=country, year=year, gdp=gdp, energy_consumption=energy_consumption, electricity_generation=electricity_generation)

    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == '__main__':
    app.run(debug=True)
