import streamlit as st

from api.backend_client import (
    check_system_health,
    run_prediction,
    get_explanation,
    check_patient_exists,
    insert_outcome,
    insert_prediction_data
)

from auth.authentication import Login
from utils.session_state import init_session_state
from utils.constants import feature_names
from user_input.ui import get_patient_payload
from utils.shap_utils import compute_shap_df
from utils.pdf_report import generate_pdf, show_pdf
from utils.logger import get_logger

logger = get_logger(__name__)

Login()

init_session_state()


# =====================================================
# App header (ALWAYS TOP)
# =====================================================
st.title("Hepatocellular Carcinoma Response Predictor")

st.markdown("""
## HCC Prediction & Outcome Tracking 🧬

This application is designed for clinicians and researchers to **predict the likelihood of success for different HCC treatment regimens** and to **track real patient outcomes**.  

**What you can do with this app:**
- Input patient **laboratory results** and **demographics**  
- Provide **tumor characteristics**, **comorbidities**, and **past medical history**  
- Include **free-text clinical notes** for additional context  
- Generate a **treatment success prediction** using a machine learning model  
- View **SHAP feature contributions** to understand which patient features influenced the prediction  
- Predict **toxicity risk** for various organ systems  
- Upload **actual patient outcomes** to the database to track whether predicted treatment plans were successful  

**How it works:**
1. Enter patient information in the sidebar.  
2. Click **Run Prediction** to get:  
   - Probability of treatment success  
   - Binary prediction (success/failure)  
   - SHAP-based feature contributions  
   - Predicted toxicity probabilities per organ system  
3. After treatment, use the **Upload Outcome** section to save the real-world outcome linked to the patient's ID.  

*Inference, explanations, and database operations are handled by backend services via secure API endpoints.*
""")

if "system_status" not in st.session_state:
    try:
        st.session_state.system_status = check_system_health()
    except Exception as e:
        st.session_state.system_status = {
            "status": "unhealthy",
            "model_loaded": False,
            "database_connected": False,
            "db_error": str(e)
        }

status = st.session_state.system_status

# Model status
if status.get("model_loaded"):
    st.success("🟢 Model loaded and ready for predictions")
else:
    st.error("🔴 Model not loaded — predictions unavailable")

# Database status
if status.get("database_connected"):
    st.success("🟢 Database connected")
else:
    st.error(f"🔴 Database connection failed: {status.get('db_error', 'Unknown error')}")

# Overall system status
if status.get("status") != "ok":
    st.warning("⚠️ System is not fully ready. Some features may be disabled.")
    st.stop()  # Stop rendering the rest




# =====================================================
# Upload Patient Outcome (Standalone Section)
# =====================================================
st.markdown("---")
st.subheader("Upload Patient Outcome to Database 📝")

st.markdown("""
This section allows clinicians or researchers to **upload treatment outcomes** for patients whose HCC predictions have already been generated.  

**Purpose:**  
- Track whether a predicted treatment plan was successful or not.  
- Link outcomes to the correct patient via a **unique Patient ID**.  
- Keep your database up-to-date for future analysis or model retraining.  

**Requirements:**  
1. **Patient ID:** Must match an existing patient in the database (from previous predictions).  
2. **Outcome:** Select whether the treatment plan was successful (`Yes`) or not (`No`).  

**How it works:**  
- The app will first **check if the Patient ID exists** in the predictions table.  
- If it exists, the outcome is uploaded to the `outcomes` table in your database.  
- If the Patient ID is not found, the upload will not proceed and you will be prompted to check the ID.  
""")

# Patient identifier
patient_id = st.text_input(
    "Patient ID",
    placeholder="Enter unique patient identifier",
    help="Must match a Patient ID already in the predictions table"
)

# Outcome selection
outcome = st.selectbox(
    "Was the treatment plan successful?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

if st.button("Check Patient and Save Outcome"):
    if not patient_id:
        st.warning("Please enter a Patient ID")
    else:
        with st.spinner("Checking if patient exists..."):
            try:
                # Check if patient exists
                patient_info = check_patient_exists(patient_id)
                if patient_info.get("exists", False):
                    st.success("✅ Patient found in database — ready to save outcome")

                    # Save outcome
                    outcome_resp = insert_outcome(
                        patient_id=patient_id,
                        outcome=outcome  # 1 = success, 0 = failure
                    )
                    st.success("✅ Outcome successfully saved")
                else:
                    st.error("❌ Patient ID not found in predictions database")

            except Exception as e:
                st.error(f"❌ Connection error or API failure: {e}")

# =====================================================
# Sidebar inputs
# =====================================================

# Get payload from sidebar inputs
payload = get_patient_payload()

# =====================================================
# Run prediction
# =====================================================
# Sidebar button to run prediction
if st.sidebar.button("Run Prediction"):
    logger.info(f"Run Prediction clicked for patient: {payload.get('patient_id')}")
    with st.spinner("Running model inference..."):
        try:
            result = run_prediction(payload)
            st.session_state.result = result
            st.session_state.has_prediction = True
            st.session_state.explanation_text = None
            st.session_state.explanation_requested = False
            logger.info("Prediction stored in session successfully")
            st.success("Prediction complete")
        except Exception as e:
            logger.error(f"Prediction failed for patient {payload.get('patient_id')}: {e}") 
            st.error(f"Prediction failed: {e}")
            st.stop()


if st.session_state.get("has_prediction"):
    result = st.session_state.result

    # ===========================
    # Model Prediction Display
    # ===========================
    st.markdown("---")
    st.subheader("Model Prediction")

    prediction_text = (
        f"The model predicted that the immunotherapy regimen would be "
        f"{'successful' if result['prediction'] == 1 else 'NOT successful'} "
        f"(predicted probability: {result['probability']:.2%})"
    )
    st.write(prediction_text)

    # ===========================
    # SHAP Feature Contributions
    # ===========================
    st.markdown("Feature contributions (SHAP):")
    try:
        shap_df = compute_shap_df(result["shap_values"], feature_names)
        st.bar_chart(shap_df.set_index("Feature"))

    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")
        shap_df = None  # for PDF fallback

    # ===========================
    # Toxicity Probabilities
    # ===========================
    st.markdown("---")
    st.subheader("Predicted Toxicity Probabilities ⚠️")
    organ_systems = [
        'skin', 'liver', 'gi', 'lung', 'endocrine',
        'neurologic', 'oral', 'musculoskeletal', 'renal', 'other'
    ]
    toxicity_probs = result.get("toxicity_proba", [0]*len(organ_systems))
    threshold = 0.05
    organs_with_risk = [
        (o.capitalize(), p) for o, p in zip(organ_systems, toxicity_probs)
        if p > threshold and o != "other"
    ]
    if organs_with_risk:
        st.write("Organ systems with higher predicted risk:")
        for organ, prob in organs_with_risk:
            st.write(f"- **{organ}**")
    else:
        st.write("No meaningful predicted risk detected.")

    # ===========================
    # Model Explanation
    # ===========================
    st.markdown("---")
    st.subheader("Model Explanation")
    if st.button("Provide explanation") and not st.session_state.get("explanation_requested", False):
        st.session_state.explanation_requested = True
        with st.spinner("Generating explanation..."):
            try:
                st.session_state.explanation_text = get_explanation(
                    probability=round(result["probability"], 2),
                    predicted_class=result["prediction"],
                    shap_values=result["shap_values"]
                )
            except Exception as e:
                st.error(f"Explanation failed: {e}")

    if st.session_state.get("explanation_text"):
        st.write(st.session_state.explanation_text)

    # ===========================
    # Save Prediction to DB
    # ===========================
    st.markdown("---")
    st.subheader("Save Prediction to Database")

    if st.button("Save Prediction to DB"):
        with st.spinner("Saving to database..."):
            try:
                # Extract flags from NER or input
                data_array = result["data"][0]
                ner_flags = {
                    "liver_tumor_flag": int(data_array[10]),
                    "liver_disease_flag": int(data_array[11]),
                    "portal_hypertension_flag": int(data_array[12]),
                    "biliary_flag": int(data_array[13]),
                    "symptoms_flag": int(data_array[14])
                }

                # Model prediction results
                model_results = {
                    "prediction": result["prediction"],
                    "probability": round(result["probability"], 2)
                }

                # Merge payload with patient info
                db_payload = {k: v for k, v in payload.items() if k != "clinical_notes"}
                db_payload.update(ner_flags)
                db_payload.update(model_results)

                # 1️⃣ Save prediction to DB
                prediction_resp = insert_prediction_data(db_payload)  # Returns inserted prediction info
                prediction_id = prediction_resp.get("id")  # Optional: if you need it later

                st.success("✅ Prediction and outcome saved to database successfully")
            except Exception as e:
                st.error(f"❌ Error saving to database: {e}")

    # ===========================
    # Generate PDF Report
    # ===========================
    st.markdown("---")
    st.subheader("Generate Summary Report")

    if st.button("Generate PDF Report"):
        try:
            pdf_file = generate_pdf(
                payload.get("patient_id"),
                result,
                st.session_state.get("explanation_text"),
                shap_df
            )

            st.session_state.generated_pdf = pdf_file

        except Exception as e:
            st.error(f"PDF generation failed: {e}")

    # Show PDF if it exists
    if "generated_pdf" in st.session_state:

        st.download_button(
            label="Download PDF",
            data=st.session_state.generated_pdf,
            file_name=f"HCC_report_{payload.get('patient_id')}.pdf",
            mime="application/pdf"
        )

        show_pdf(st.session_state.generated_pdf)