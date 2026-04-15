from fastapi import APIRouter, HTTPException

from schemas.patient import PatientData
from schemas.Results import ResultsData
from schemas.db import DB_Data
from schemas.Patientcheck import Patient_check
from schemas.Outcome import OutcomeCreate

from services.model_service import ModelService
from services.Feature_service import build_features
from services.explanation_service import ExplanationService

import traceback
import os

from psycopg2 import pool
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path, override=True)

db_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    sslmode=os.getenv("DB_SSLMODE", "require")
)


router = APIRouter()

model_service = ModelService()
explanation_service = ExplanationService()

@router.post("/predict")
def predict(patient: PatientData):
    """
    Predict HCC response probability, SHAP feature contributions, and toxicity
    """
    try:
        # Build features in proper order
        X = build_features(patient).reshape(1, -1)

        # HCC prediction
        proba = model_service.hcc_predict(X)
        prediction = int(proba[0, 1] > 0.5)  # binary threshold

        # SHAP explanation
        shap_values, base_value = model_service.explain_prediction(X)

        # Toxicity prediction
        toxicity_proba = model_service.predict_toxicity(X)
        
        return {
            "probability": float(proba[0, 1]),
            "prediction": prediction,
            "shap_values": shap_values.tolist(),  # SHAP values per feature
            "baseline": base_value,                         # base value / expected value
            "toxicity_proba": toxicity_proba[0].toarray().flatten().tolist(),
            "data": X.tolist()
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/explanation')
def explain(results: ResultsData):
    shap_values_list = results.shap_values
    probability = results.probability

    explanation = explanation_service.generate_explanation(probability, shap_values_list)
    return {'explanation': explanation}


@router.get('/health')
def health_check():
    # Check if the model is loaded
    model_status = model_service is not None

    # Check database connection
    db_status = False
    db_error = None
    conn = None
    try:
       conn = db_pool.getconn()
       cur = conn.cursor()
       
       cur.execute('SELECT 1;')
       cur.fetchone()

       db_status = True
       cur.close()

    except Exception as e:
        db_error = str(e)

    finally:
        if conn:
            db_pool.putconn(conn)

    return {
        'status': 'ok' if model_status and db_status else 'unhealthy',
        'model_loaded': model_status,
        'database_connected': db_status,
        'db_error': db_error
    }
   
@router.post("/insert_data")
def insert_data(db_info: DB_Data):

    conn = None

    try:
        conn = db_pool.getconn()
        cur = conn.cursor()

        query = """
        INSERT INTO "HCC_Predictions" (
            patient_id,
            ast, alt, alp, albumin, total_bilirubin, afp,
            stage_at_diagnosis, t_stage_at_diagnosis,
            age, gender,
            pmh_cirrhosis, pmh_fatty_liver, comorbid_diabetes, comorbid_htn, comorbid_cad,
            regimen_atezo_bev, regimen_durva_treme, regimen_nivo_ipi, regimen_pembro_ipi,
            local_treatment_given_TACE, local_treatment_given_Y90,
            local_treatment_given_RFA, local_treatment_given_None,
            neoadjuvant_therapy, adjuvant_treatment_given,
            liver_tumor_flag, liver_disease_flag, portal_hypertension_flag,
            biliary_flag, symptoms_flag,
            prediction, probability
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s,
            %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s
        )
        ON CONFLICT (patient_id) DO NOTHING
        """

        cur.execute(query, (
            db_info.patient_id,
            db_info.ast, db_info.alt, db_info.alp,
            db_info.albumin, db_info.total_bilirubin, db_info.afp,
            db_info.stage_at_diagnosis, db_info.t_stage_at_diagnosis,
            db_info.age, db_info.gender,
            db_info.pmh_cirrhosis, db_info.pmh_fatty_liver,
            db_info.comorbid_diabetes, db_info.comorbid_htn,
            db_info.comorbid_cad,
            db_info.regimen_atezo_bev, db_info.regimen_durva_treme,
            db_info.regimen_nivo_ipi, db_info.regimen_pembro_ipi,
            db_info.local_treatment_given_TACE,
            db_info.local_treatment_given_Y90,
            db_info.local_treatment_given_RFA,
            db_info.local_treatment_given_None,
            db_info.neoadjuvant_therapy,
            db_info.adjuvant_treatment_given,
            db_info.liver_tumor_flag,
            db_info.liver_disease_flag,
            db_info.portal_hypertension_flag,
            db_info.biliary_flag,
            db_info.symptoms_flag,
            db_info.prediction,
            db_info.probability
        ))

        conn.commit()
        cur.close()

        return {"status": "success"}

    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        if conn:
            db_pool.putconn(conn)

@router.post("/check_data")
def check_data(data: Patient_check):

    conn = None

    try:
        conn = db_pool.getconn()
        cur = conn.cursor()

        query = """
        SELECT 1
        FROM "HCC_Predictions"
        WHERE patient_id = %s
        LIMIT 1
        """

        cur.execute(query, (data.patient_id,))
        result = cur.fetchone()

        exists = result is not None

        cur.close()

        return {
            "exists": exists,
            "patient_id": data.patient_id
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database check failed: {str(e)}"
        )

    finally:
        if conn:
            db_pool.putconn(conn)
    
@router.post("/insert_outcome")
def insert_outcome(data: OutcomeCreate):
    conn = None
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()

        # Look up prediction_id from patient_id
        cur.execute(
            'SELECT id FROM "HCC_Predictions" WHERE patient_id = %s LIMIT 1',
            (data.patient_id,)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Patient ID not found")

        prediction_id = row[0]

        # Insert outcome using the prediction_id
        cur.execute(
            """
            INSERT INTO "Outcomes" (prediction_id, outcome)
            VALUES (%s, %s)
            RETURNING id, prediction_id, outcome, created_at
            """,
            (prediction_id, data.outcome)
        )
        conn.commit()
        inserted_row = cur.fetchone()
        cur.close()

        return {
            "status": "success",
            "inserted": {
                "id": inserted_row[0],
                "prediction_id": inserted_row[1],
                "outcome": inserted_row[2],
                "created_at": inserted_row[3].isoformat()
            }
        }

    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database insert failed: {str(e)}")
    finally:
        if conn:
            db_pool.putconn(conn)