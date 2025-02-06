#!/usr/bin/env python
# coding: utf-8

# This script is adapted from the notebooks "./notebook/part_1_preprocessing.ipynb"
# and "./notebook/part_2_modeling.ipynb".
# Only the final model training part is preserved as per the ML-Zoomcamp midterm criteria below:
# ==============================================================================================
# * Script train.py (suggested name)
#   * Training the final model.
#   * Saving it to a file (e.g. pickle) or saving it with specialized software (BentoML). 
# ==============================================================================================


import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.feature_extraction import DictVectorizer

# ### Potential issue with loading XGBoost
# 
# There might be potential issues with importing `xgboost` in the following code block due to your local Operating System. 
# 
# For example, this notebook was ran on Mac OS X and required additional setup in terminal: `brew install libomp` to fix the import issue.
# 
# Please follow the suggestions given in the error message and fix accordingly.
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
    
    # Drop the only record with RestingBP = 0
    df = df[df['RestingBP'] > 0]

    # Transform to categorical data type
    category_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for c in category_cols:
        df[c] = df[c].astype("category")
    
    # Replace invalid cholesterol data with median value
    median_cholesterol = df.loc[df['Cholesterol'] > 0, 'Cholesterol'].median()
    df.loc[df['Cholesterol'] == 0, 'Cholesterol'] = int(median_cholesterol)

    # Add new feature Cholesterol_Level
    df['Cholesterol_Level'] = df['Cholesterol'].map(get_cholesterol_level)
    # Returns cleaned dataset
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

# ==============================================================================================
# STEPS:
# ==============================================================================================
# * Data preparation:
#     * Clean up dataset.
#     * Feature engineering.  
#     * Split data into train/validation/test 60%/20%/20%
# * Data transformation:
#     * Encode the dataset using the DictVectorizer
# * Hyperparameter-tuning to find best model parameters.
# * Train :
#     * Train an XGBoost Classifier model using the best model parameters.
#     * Predict and evaluate model on all data sets using confusion matrix, AUC scoring.
# * Train the final model and dictvectorizer using full dataset
# * Save the final model and dictvectorizer using Pickle

if __name__ == "__main__":
    # Data Preparation
    df = prepare_data(file)

    # Split data
    y = df['HeartDisease']
    X = df.drop(columns=['HeartDisease'])
    df_full_train, df_test, y_full_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=True, stratify=y)
    df_train, df_val, y_train, y_val = train_test_split(df_full_train, y_full_train, test_size=0.25, random_state=seed, shuffle=True, 
                                                        stratify=y_full_train)
    # Verify that the split sets' sizes tally with original dataset size
    assert y_train.shape[0] + y_val.shape[0] + y_test.shape[0] == df.shape[0]
    
    # Data transformation using DictVectorizer
    dv, X_train = encode_data(df_train)
    features = dv.get_feature_names_out()

    # Hyperparameter-tuning  
    best_model = finetune_model(X_train, y_train)
    print("Finetuned XGBoost Classifier parameters:") 
    print(best_model.best_params_)

    # Train the model using the finetuned parameters.
    xgb = XGBClassifier(random_state=seed, **best_model.best_params_)
    xgb.fit(X_train, y_train)

    # print confusion matrix structure for reference
    cm = np.array([['TN', 'FP'],['FN', 'TP']])
    print("Confusion Matrix reference:")
    print(cm)

    # Evaluate model using training dataset
    train_pred = predict(df_train.to_dict(orient='records'), dv, xgb)
    train_pred = np.where(train_pred >= 0.5, 1, 0)
    print(f"Training set confusion matrix:\n{confusion_matrix(y_train, train_pred)}")
    print(f"AUC score for training data: {roc_auc_score(y_train, train_pred):.4f}") 

    # Evaluate model using validation dataset
    dict_val = df_val.to_dict(orient='records')
    val_pred = predict(dict_val, dv, xgb)
    val_pred = np.where(val_pred >= 0.5, 1, 0)
    print(f"Validation set confusion matrix:\n{confusion_matrix(y_val, val_pred)}")
    print(f"AUC score for validation data: {roc_auc_score(y_val, val_pred):.4f}") 

    # Evaluate model using test dataset
    dict_test = df_test.to_dict(orient='records')
    test_pred = predict(dict_test, dv, xgb)
    test_pred = np.where(test_pred >= 0.5, 1, 0)
    print(f"Validation set confusion matrix:\n{confusion_matrix(y_test, test_pred)}")
    print(f"AUC score for test data: {roc_auc_score(y_test, test_pred):.4f}") 

    # Train the final model and DictVectorizer using full dataset
    dv_final, X_transformed = encode_data(X)
    xgb_final = XGBClassifier(random_state=seed, **best_model.best_params_)
    xgb_final.fit(X_transformed, y)

    # Save the final model and DictVectorizer using Pickle
    f_out = open(final_model, 'wb') 
    pickle.dump((dv_final, xgb_final), f_out)
    f_out.close()
    print(f"XGB Model and DictVectorizer saved to {final_model}.")
