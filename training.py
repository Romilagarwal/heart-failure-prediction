#!/usr/bin/env python
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.feature_extraction import DictVectorizer

from xgboost import XGBClassifier

file = "./data/heart.csv"
seed = 11
final_model = "heart_disease_model.bin"

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

def prepare_data(file):
    '''
    Load, clean, transform and feature engineering as per part_1_preprocessing.ipynb
    Returns a cleaned dataset
    '''
    df = pd.read_csv(file)

    df = df[df['RestingBP'] > 0]

    category_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for c in category_cols:
        df[c] = df[c].astype("category")

    median_cholesterol = df.loc[df['Cholesterol'] > 0, 'Cholesterol'].median()
    df.loc[df['Cholesterol'] == 0, 'Cholesterol'] = int(median_cholesterol)

    df['Cholesterol_Level'] = df['Cholesterol'].map(get_cholesterol_level)
    return df

def encode_data(df_train):
    '''
    Trains a DictVectorizer model using the training dataset to perform one-hot encoding.
    Returns trained DictVectorizer model, encoded training dataset
    '''
    dict_train = df_train.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dict_train)
    return dv, X_train

def finetune_model(features, target):
    '''
    Hyperparameter-tuning using Grid Search CV on XGBClassifier
    '''
    parameters = {'eta' : [0.005, 0.01, 0.05, 0.1, 1],
                'max_depth' : [2, 3, 4, 5],
                'n_estimators': [100, 200, 300],
                'min_child_weight': [1, 2, 3]
                }
    xgb = XGBClassifier(random_state=seed)
    gcv_xgb = GridSearchCV(xgb, parameters, cv=5, scoring='roc_auc', n_jobs=-1)
    gcv_xgb.fit(features, target)
    return gcv_xgb

def predict(features, dv, model):
    '''
    Transforms provided features dictionary and predicts if patient has heart disease
    Return the predicted probability of having heart disease in range [0,1].
    '''
    transformed = dv.transform(features)
    y_pred = model.predict_proba(transformed)[:,1]
    return y_pred

if __name__ == "__main__":
    df = prepare_data(file)

    y = df['HeartDisease']
    X = df.drop(columns=['HeartDisease'])
    df_full_train, df_test, y_full_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=True, stratify=y)
    df_train, df_val, y_train, y_val = train_test_split(df_full_train, y_full_train, test_size=0.25, random_state=seed, shuffle=True,
                                                        stratify=y_full_train)
    assert y_train.shape[0] + y_val.shape[0] + y_test.shape[0] == df.shape[0]

    dv, X_train = encode_data(df_train)
    features = dv.get_feature_names_out()

    best_model = finetune_model(X_train, y_train)
    print("Finetuned XGBoost Classifier parameters:")
    print(best_model.best_params_)

    xgb = XGBClassifier(random_state=seed, **best_model.best_params_)
    xgb.fit(X_train, y_train)

    cm = np.array([['TN', 'FP'],['FN', 'TP']])
    print("Confusion Matrix reference:")
    print(cm)

    train_pred = predict(df_train.to_dict(orient='records'), dv, xgb)
    train_pred = np.where(train_pred >= 0.5, 1, 0)
    print(f"Training set confusion matrix:\n{confusion_matrix(y_train, train_pred)}")
    print(f"AUC score for training data: {roc_auc_score(y_train, train_pred):.4f}")

    dict_val = df_val.to_dict(orient='records')
    val_pred = predict(dict_val, dv, xgb)
    val_pred = np.where(val_pred >= 0.5, 1, 0)
    print(f"Validation set confusion matrix:\n{confusion_matrix(y_val, val_pred)}")
    print(f"AUC score for validation data: {roc_auc_score(y_val, val_pred):.4f}")

    dict_test = df_test.to_dict(orient='records')
    test_pred = predict(dict_test, dv, xgb)
    test_pred = np.where(test_pred >= 0.5, 1, 0)
    print(f"Validation set confusion matrix:\n{confusion_matrix(y_test, test_pred)}")
    print(f"AUC score for test data: {roc_auc_score(y_test, test_pred):.4f}")

    dv_final, X_transformed = encode_data(X)
    xgb_final = XGBClassifier(random_state=seed, **best_model.best_params_)
    xgb_final.fit(X_transformed, y)

    f_out = open(final_model, 'wb')
    pickle.dump((dv_final, xgb_final), f_out)
    f_out.close()
    print(f"XGB Model and DictVectorizer saved to {final_model}.")
