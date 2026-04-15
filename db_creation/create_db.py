import psycopg2
import os 
from dotenv import load_dotenv

load_dotenv()


conn = None
try:
    conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            sslmode="verify-full",
            sslrootcert=r"C:\Users\kilsi\OneDrive\Documents\Curenetics\HCC_APP\db_creation\certs\global-bundle.pem"
        )
    cur = conn.cursor()

    create_table_query = """
    CREATE TABLE IF NOT EXISTS "HCC_Predictions" (

        id SERIAL PRIMARY KEY,

        patient_id VARCHAR(30) UNIQUE NOT NULL,

        ast DOUBLE PRECISION NOT NULL,
        alt DOUBLE PRECISION NOT NULL,
        alp DOUBLE PRECISION NOT NULL,
        albumin DOUBLE PRECISION NOT NULL,
        total_bilirubin DOUBLE PRECISION NOT NULL,
        afp DOUBLE PRECISION NOT NULL,

        stage_at_diagnosis INTEGER NOT NULL,
        t_stage_at_diagnosis INTEGER NOT NULL,

        age INTEGER NOT NULL,
        gender INTEGER NOT NULL CHECK (gender IN (0,1)),

        pmh_cirrhosis INTEGER NOT NULL CHECK (pmh_cirrhosis IN (0,1)),
        pmh_fatty_liver INTEGER NOT NULL CHECK (pmh_fatty_liver IN (0,1)),
        comorbid_diabetes INTEGER NOT NULL CHECK (comorbid_diabetes IN (0,1)),
        comorbid_htn INTEGER NOT NULL CHECK (comorbid_htn IN (0,1)),
        comorbid_cad INTEGER NOT NULL CHECK (comorbid_cad IN (0,1)),

        regimen_atezo_bev INTEGER NOT NULL CHECK (regimen_atezo_bev IN (0,1)),
        regimen_durva_treme INTEGER NOT NULL CHECK (regimen_durva_treme IN (0,1)),
        regimen_nivo_ipi INTEGER NOT NULL CHECK (regimen_nivo_ipi IN (0,1)),
        regimen_pembro_ipi INTEGER NOT NULL CHECK (regimen_pembro_ipi IN (0,1)),

        local_treatment_given_TACE INTEGER NOT NULL CHECK (local_treatment_given_TACE IN (0,1)),
        local_treatment_given_Y90 INTEGER NOT NULL CHECK (local_treatment_given_Y90 IN (0,1)),
        local_treatment_given_RFA INTEGER NOT NULL CHECK (local_treatment_given_RFA IN (0,1)),
        local_treatment_given_None INTEGER NOT NULL CHECK (local_treatment_given_None IN (0,1)),

        neoadjuvant_therapy INTEGER NOT NULL CHECK (neoadjuvant_therapy IN (0,1)),
        adjuvant_treatment_given INTEGER NOT NULL CHECK (adjuvant_treatment_given IN (0,1)),

        liver_tumor_flag INTEGER NOT NULL CHECK (liver_tumor_flag IN (0,1)),
        liver_disease_flag INTEGER NOT NULL CHECK (liver_disease_flag IN (0,1)),
        portal_hypertension_flag INTEGER NOT NULL CHECK (portal_hypertension_flag IN (0,1)),
        biliary_flag INTEGER NOT NULL CHECK (biliary_flag IN (0,1)),
        symptoms_flag INTEGER NOT NULL CHECK (symptoms_flag IN (0,1)),

        prediction INTEGER NOT NULL CHECK (prediction IN (0,1)),
        probability DOUBLE PRECISION NOT NULL,

        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """

    create_table_query1 = """
        CREATE TABLE IF NOT EXISTS "Outcomes" (
            id SERIAL PRIMARY KEY,

            prediction_id INTEGER NOT NULL,

            outcome INTEGER NOT NULL CHECK (outcome IN (0,1)),

            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

            CONSTRAINT fk_prediction
                FOREIGN KEY (prediction_id)
                REFERENCES "HCC_Predictions"(id)
                ON DELETE CASCADE
        );
        """

    cur.execute(create_table_query)
    print("Table HCC_Predictions created (or already exists).")

    cur.execute(create_table_query1)
    print("Outcomes table created (or already exists).")

    conn.commit()

    
    cur.close()
except Exception as e:
    print(f"Database error: {e}")
    raise
finally:
    if conn:
        conn.close()


