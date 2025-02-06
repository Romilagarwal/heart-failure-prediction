import pickle

from flask import Flask
from flask import request # Process incoming request via POST
from flask import jsonify # Send back response in JSON format


model_file = 'heart_disease_model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('heart_disease_app')

## use POST method to send the patient information
@app.route('/predict_risk', methods=['POST'])
def predict():
    '''
    Transforms provided patient info and predicts if patient has heart disease
    Return the predicted probability and boolean value of having heart disease in range [0,1].
    '''
    patient_info = request.get_json()
    transformed = dv.transform(patient_info)
    y_pred = model.predict_proba(transformed)[:,1]
    hasHeartDisease = y_pred >= 0.5
    result = {
        'hasHeartDisease_probability': float(y_pred[0]),
        'hasHeartDisease': bool(hasHeartDisease)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)