from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('insurance_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    age = int(request.form['age'])

    is_female = int(request.form['is_female'])

    bmi = float(request.form['bmi'])

    children = int(request.form['children'])

    is_smoker = int(request.form['is_smoker'])

    region_southeast = int(
        request.form['region_southeast']
    )

    bmi_category_obese = int(
        request.form['bmi_category_Obese']
    )

    input_data = np.array([[
        age,
        is_female,
        bmi,
        children,
        is_smoker,
        region_southeast,
        bmi_category_obese
    ]])

    prediction = model.predict(input_data)[0]

    return render_template(
        'index.html',
        prediction_text=f'Estimated Insurance Cost: ${prediction:.2f}'
    )

if __name__ == '__main__':
    app.run(debug=True)