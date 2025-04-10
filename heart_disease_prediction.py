import streamlit as st
import pickle

model_file = 'heart_disease_model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

st.title("Heart Disease Prediction app")
st.markdown("")
st.caption("Disclaimer: The predictions are not meant to be used in a professional medical setting. Please consult a medical professional if required.")
st.subheader("Enter patient particulars below")

with st.form("patient_info"):
    age = st.number_input("Age", min_value=1, max_value=100)

    sex_list = {"M":"Male", "F":"Female"}
    sex = st.radio("Sex", options=sex_list.keys(), format_func=lambda x: sex_list.get(x),
                index=None)

    chest_list = {"TA": "Typical Angina", "ATA": "Atypical Angina",
                "NAP": "Non-Anginal Pain", "ASY": "Asymptomatic"}
    chestPainType = st.radio("Chest Pain Type", options=chest_list.keys(),
                            format_func=lambda x: chest_list.get(x), index=None)
    restingBP = st.number_input("Resting Blood Pressure (mm/Hg)", min_value=1)
    cholesterol = st.number_input("Cholesterol (mm/dl)", min_value=1)

    bs_list = {1: "Fasting Blood Sugar above 120 mg/dl", 0: "Otherwise"}
    fastingBS = st.radio("Fasting Blood Sugar()", options=bs_list.keys(),
                            format_func=lambda x: bs_list.get(x), index=None)
    ecg_list = {"Normal": "Normal",
                "ST": "having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)",
                "LVH": "showing probable or definite left ventricular hypertrophy by Estes' criteria]"
                }
    restingECG = st.radio("Resting Electrocardiogram Results", options=ecg_list.keys(),
                            format_func=lambda x: ecg_list.get(x), index=None)

    maxHR = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=202)

    angina_yn = {"Y": "Yes", "N": "No"}
    exerciseAngina = st.radio("Exercise-induced Angina", options=angina_yn.keys(),
                            format_func=lambda x: angina_yn.get(x), index=None)

    oldpeak = st.number_input("Oldpeak=ST (numeric value measured in depression)", format="%0.1f")

    st_list = {"Up": "Upsloping", "Flat": "Flat", "Down": "Downsloping"}
    st_slope = st.radio("ST Slope", options=st_list.keys(),
                            format_func=lambda x: st_list.get(x), index=None)
    submit = st.form_submit_button("Submit patient info and predict heart disease")

def get_cholesterol_level(c):
    '''
    Feature engineering on cholesterol_level
    '''
    if c < 200:
        return 'Normal'
    elif c <= 239:
        return 'Borderline high'
    else:
        return 'High'

def predict(features, dv, model):
    '''
    Transforms provided features dictionary and predicts if patient has heart disease
    Return the predicted probability of having heart disease in range [0,1].
    '''
    transformed = dv.transform(features)
    y_pred = model.predict_proba(transformed)[:,1] >= 0.5
    return y_pred

if submit:
    patient_info = {"Age": age,
                    "Sex": sex,
                    "ChestPainType": chestPainType,
                    "RestingBP": restingBP,
                    "Cholesterol": cholesterol,
                    "FastingBS": fastingBS,
                    "RestingECG": restingECG,
                    "MaxHR": maxHR,
                    "ExerciseAngina": exerciseAngina,
                    "Oldpeak": oldpeak,
                    "ST_Slope": st_slope,
                    "Cholesterol_Level": get_cholesterol_level(cholesterol)
                    }

    hasHeartDisease = predict(patient_info, dv, model)
    st.subheader("Prediction Results:")
    if hasHeartDisease:
        st.write("Potentially at risk of heart disease. Follow-up examination recommended.")
    else:
        st.write("Negative.")
