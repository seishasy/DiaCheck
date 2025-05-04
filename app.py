from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Mapping categorical inputs
smoking_mapping = {
    "No Info": 0,
    "never": 0,
    "former": 1,
    "ever": 1,
    "current": 1,
    "not current": 1
}

gender_mapping = {
    "Male": 0,
    "Female": 1,
    "Other": 1
}


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict-form')
def predictForm():
    return render_template('pages/form-diabetes.html')

@app.route('/predict-score', methods=['POST'])
def predictScore():
    try:
        gender = gender_mapping.get(request.form['gender'].lower(), 0)
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        smoking_history = smoking_mapping.get(request.form['smoking_history'].lower(), 0)
        bmi = float(request.form['bmi'])
        HbA1c_level = float(request.form['hba1c_level'])
        blood_glucose_level = float(request.form['blood_glucose_level'])
        
        # Data input dalam bentuk array numpy
        input_data = np.array([[
            gender, age, hypertension, heart_disease, smoking_history,
            bmi, HbA1c_level, blood_glucose_level
        ]])
        # Prediksi model
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1] * 100
        # accuracy = accuracy_score(prediction)
        result = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes"
        score = probability
        return render_template('pages/predict-diabetes.html', result=result, score=score)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)