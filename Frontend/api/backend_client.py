import requests
from config.settings import BACKEND_URL
from utils.logger import get_logger


logger = get_logger(__name__)
# --------------------------------------------------
# Health check
# --------------------------------------------------
def check_system_health():

    logger.info("Health check started")

    try:
        r = requests.get(
            f"{BACKEND_URL}/api/v1/health",
            timeout=60
        )

        logger.info(f"Health check response: {r.status_code}")

        r.raise_for_status()
        result = r.json()

        logger.info(f"Model loaded: {result.get('model_loaded')} | DB connected: {result.get('database_connected')}")

        return result

    except requests.exceptions.RequestException as e:

        logger.error(f"Health check failed: {e}")

        return {
            "status": "unhealthy",
            "model_loaded": False,
            "database_connected": False,
            "db_error": str(e)
        }


# --------------------------------------------------
# Run prediction
# --------------------------------------------------
def run_prediction(payload):

    logger.info(f"Sending prediction request for patient: {payload.get('patient_id')}")

    try:

        r = requests.post(
            f"{BACKEND_URL}/api/v1/predict",
            json=payload,
            timeout=120
        )

        logger.info(f"Prediction response status: {r.status_code}")

        r.raise_for_status()

        result = r.json()

        logger.info(f"Prediction success: probability={result.get('probability')}, prediction={result.get('prediction')}")

        return result 
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Prediction request failed: {e}")
        raise
        


# --------------------------------------------------
# Generate explanation
# --------------------------------------------------
def get_explanation(probability, predicted_class, shap_values):

    payload = {
        "probability": probability,
        "predicted_class": predicted_class,
        "shap_values": shap_values
    }

    logger.info(f'Sending explanation request: probability={probability}')

    try:

        r = requests.post(
            f"{BACKEND_URL}/api/v1/explanation",
            json=payload,
            timeout=180
        )

        logger.info(f"Explanation response status: {r.status_code}")

        r.raise_for_status()

        result = r.json()

        logger.info("Explanation received successfully")

        return result["explanation"]
    
    except requests.exceptions.RequestException as e:

        logger.error(f"explanation request failed: {e}")
        raise


# --------------------------------------------------
# Check if patient exists
# --------------------------------------------------
def check_patient_exists(patient_id):

    payload = {"patient_id": patient_id}

    logger.info(f'Sending request to check if patient {patient_id} is in the database')

    try:

        r = requests.post(
            f"{BACKEND_URL}/api/v1/check_data",
            json=payload,
            timeout=60
        )

        logger.info(f"check response status: {r.status_code}")

        r.raise_for_status()

        result =  r.json()

        logger.info(f"Check received successfully — exists: {result.get('exists')}")

        return result 

    except requests.exceptions.RequestException as e:
        logger.error(f"check request failed: {e}")
        raise 



# --------------------------------------------------
# Save patient outcome
# --------------------------------------------------
def insert_outcome(patient_id, outcome):

    payload = {
        "patient_id": patient_id,
        "outcome": outcome
    }

    logger.info(f'Insert outcome for patient: {patient_id}')

    try:

        r = requests.post(
            f"{BACKEND_URL}/api/v1/insert_outcome",
            json=payload,
            timeout=60
        )

        logger.info(f'checking response status: {r.status_code}')

        r.raise_for_status()

        result = r.json()

        logger.info(f"Outcome inserted successfully for patient: {patient_id}")

        return result
    
    except requests.exceptions.RequestException as e:
        logger.error(f'Could not insert patient: {patient_id} outcome into the database: {e}')
        raise 


# --------------------------------------------------
# Save prediction data
# --------------------------------------------------

def insert_prediction_data(payload):
    """
    Sends prediction data to the backend API and returns the inserted row info.
    """
    logger.info(f"Inserting prediction into database for patient: {payload.get('patient_id')}")

    try:
        r = requests.post(
            f"{BACKEND_URL}/api/v1/insert_data",
            json=payload,
            timeout=60
        )

        logger.info(f"Insert prediction response status: {r.status_code}")  # ← added

        r.raise_for_status()

        result = r.json()

        logger.info(f"Successfully inserted prediction into database")  # ← moved after r.json()

        return result

    except requests.exceptions.RequestException as e:
        logger.error(f"Could not insert prediction into database: {e}")
        raise
