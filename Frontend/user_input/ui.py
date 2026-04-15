import streamlit as st 
from utils.encoders import stage_dict, t_stage_dict

def get_patient_payload():
    """Collects user inputs from sidebar and builds API payload."""
    
    # --------------------------
    # Sidebar Inputs
    # --------------------------
    st.sidebar.header("Patient Inputs")
    patient_id = st.sidebar.text_input("Patient ID", placeholder="Enter unique patient identifier")

    # Demographics
    age = st.sidebar.number_input("Age", 0, 120, 60)
    gender = st.sidebar.selectbox("Sex at Birth", ["Male", "Female"])

    # Laboratory Results
    ast = st.sidebar.number_input("AST (U/L)", 0, 1000, 30)
    alt = st.sidebar.number_input("ALT (U/L)", 0, 1000, 25)
    alp = st.sidebar.number_input("ALP (U/L)", 0, 1000, 80)
    albumin = st.sidebar.number_input("Albumin (g/L)", 0.0, 1000.0, 40.0)
    total_bilirubin = st.sidebar.number_input("Total Bilirubin (mg/dL)", 0.0, 1000.0, 1.0)
    afp = st.sidebar.number_input("AFP (ng/mL)", 0.0, 100000.0, 10.0)

    # Tumour Characteristics
    stage = st.sidebar.selectbox("Stage at Diagnosis", list(stage_dict.keys()))
    t_stage = st.sidebar.selectbox("T-stage", list(t_stage_dict.keys()))

    # Systemic Therapy
    systemic_regimen = st.sidebar.selectbox(
        "Systemic Regimen",
        ["None", "Atezolizumab + Bevacizumab", "Durvalumab + Tremelimumab",
         "Nivolumab + Ipilimumab", "Pembrolizumab + Ipilimumab"]
    )

    # Local Liver Treatment
    local_treatment = st.sidebar.selectbox("Local Treatment", ["None", "TACE", "Y90", "RFA"])

    # Comorbidities / History
    cirrhosis = st.sidebar.selectbox("Cirrhosis", ["No", "Yes"])
    fatty_liver = st.sidebar.selectbox("Fatty Liver Disease", ["No", "Yes"])
    diabetes = st.sidebar.selectbox("Diabetes", ["No", "Yes"])
    htn = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
    cad = st.sidebar.selectbox("Coronary Artery Disease", ["No", "Yes"])

    # Other Treatments
    neoadjuvant = st.sidebar.selectbox("Neoadjuvant Therapy Given?", ["No", "Yes"])
    adjuvant = st.sidebar.selectbox("Adjuvant Therapy Given?", ["No", "Yes"])

    # Clinical Notes
    clinical_notes = st.sidebar.text_area("Clinical Notes", placeholder="Type patient notes here...")

    # --------------------------
    # Encode inputs
    # --------------------------
    gender_encoded = 1 if gender == "Male" else 0

    regimen_atezo_bev = int(systemic_regimen == "Atezolizumab + Bevacizumab")
    regimen_durva_treme = int(systemic_regimen == "Durvalumab + Tremelimumab")
    regimen_nivo_ipi = int(systemic_regimen == "Nivolumab + Ipilimumab")
    regimen_pembro_ipi = int(systemic_regimen == "Pembrolizumab + Ipilimumab")

    local_TACE = int(local_treatment == "TACE")
    local_Y90 = int(local_treatment == "Y90")
    local_RFA = int(local_treatment == "RFA")
    local_None = int(local_treatment == "None")

    # --------------------------
    # Build payload
    # --------------------------
    payload = {
        "patient_id": patient_id,
        "age": int(age),
        "gender": gender_encoded,
        "ast": ast,
        "alt": alt,
        "alp": alp,
        "albumin": albumin,
        "total_bilirubin": total_bilirubin,
        "afp": afp,
        "stage_at_diagnosis": stage_dict[stage],
        "t_stage_at_diagnosis": t_stage_dict[t_stage],
        "pmh_cirrhosis": int(cirrhosis == "Yes"),
        "pmh_fatty_liver": int(fatty_liver == "Yes"),
        "comorbid_diabetes": int(diabetes == "Yes"),
        "comorbid_htn": int(htn == "Yes"),
        "comorbid_cad": int(cad == "Yes"),
        "regimen_atezo_bev": regimen_atezo_bev,
        "regimen_durva_treme": regimen_durva_treme,
        "regimen_nivo_ipi": regimen_nivo_ipi,
        "regimen_pembro_ipi": regimen_pembro_ipi,
        "local_treatment_given_TACE": local_TACE,
        "local_treatment_given_Y90": local_Y90,
        "local_treatment_given_RFA": local_RFA,
        "local_treatment_given_None": local_None,
        "neoadjuvant_therapy": int(neoadjuvant == "Yes"),
        "adjuvant_treatment_given": int(adjuvant == "Yes"),
        "clinical_notes": clinical_notes
    }

    return payload