import numpy as np
import shap
import pandas as pd
import os
import boto3
from dotenv import load_dotenv
import pickle

load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

# Bucket and object keys
bucket_name = "hcc-app-model-weights-2026"
model_keys = {
    "hcc_model": "hcc_model.pkl",
    "toxicity_model": "toxicity_model.pkl"
}

# Function to load pickle object from S3
def load_pickle_from_s3(bucket: str, key: str):
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj['Body'].read()
    return pickle.loads(data)

FEATURE_NAMES = [
    'ast', 'alt', 'alp', 'albumin', 'total_bilirubin', 'afp',
    'stage_at_diagnosis', 't_stage_at_diagnosis', 'age', 'gender',
    'pmh_cirrhosis', 'pmh_fatty_liver', 'comorbid_diabetes',
    'comorbid_htn', 'comorbid_cad',
    'liver_tumor_flag', 'liver_disease_flag', 'portal_hypertension_flag',
    'biliary_flag', 'symptoms_flag',
    'regimen_atezo_bev', 'regimen_durva_treme',
    'regimen_nivo_ipi', 'regimen_pembro_ipi',
    'local_treatment_given_TACE', 'local_treatment_given_Y90',
    'local_treatment_given_RFA', 'local_treatment_given_None',
    'neoadjuvant_therapy', 'adjuvant_treatment_given'
]




class ModelService:
    def __init__(self):
        # Load models
        self.hcc_model = load_pickle_from_s3(bucket_name, "hcc_model.pkl")
        self.toxicity_model = load_pickle_from_s3(bucket_name, "toxicity_model.pkl")
        
        # SHAP explainer for the RandomForest
        self.explainer = shap.TreeExplainer(self.hcc_model)

    def _to_df(self, X):
        # Ensure X is a DataFrame with proper feature names
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return pd.DataFrame(X, columns=FEATURE_NAMES)

    def hcc_predict(self, X):
        X_df = self._to_df(X)
        return self.hcc_model.predict_proba(X_df)

    def explain_prediction(self, X):
        shap_values = self.explainer.shap_values(X)
        base_value = self.explainer.expected_value[1]  # positive class
        print(shap_values)
        return shap_values[0,:,1], base_value  # first patient, class 1

    def predict_toxicity(self, X):
        """
        Returns predicted toxicity probabilities for each organ system.
        """
        X_df = self._to_df(X)
        return self.toxicity_model.predict_proba(X_df)
