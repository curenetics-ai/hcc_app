import pandas as pd
import numpy as np

def compute_shap_df(shap_values, feature_names, top_n=None):
    """
    Convert SHAP values into a DataFrame sorted by absolute importance.

    Parameters:
    - shap_values: array-like, SHAP values from model
    - feature_names: list of feature names
    - top_n: optional, number of top features to keep

    Returns:
    - shap_df: pd.DataFrame with 'Feature' and 'SHAP value' columns
    """
    shap_array = np.array(shap_values)
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP value": shap_array
    }).sort_values(by="SHAP value", key=abs, ascending=False)

    if top_n:
        shap_df = shap_df.head(top_n)

    return shap_df