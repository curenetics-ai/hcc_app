# utils/pdf_report.py

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from io import BytesIO
import datetime
import matplotlib.pyplot as plt
import numpy as np
import base64
import streamlit as st

def create_shap_plot(shap_df):
    fig, ax = plt.subplots(figsize=(6,4))
    shap_df_top = shap_df.head(10)
    ax.barh(shap_df_top["Feature"], shap_df_top["SHAP value"])
    ax.set_xlabel("SHAP Value")
    ax.set_title("Top Feature Contributions")
    plt.tight_layout()
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format="png")
    plt.close()
    img_buffer.seek(0)
    return img_buffer

def generate_pdf(patient_id, result, explanation_text, shap_df):
    buffer = BytesIO()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    elements.append(Paragraph("Hepatocellular Carcinoma Prediction Report", styles['Title']))
    elements.append(Spacer(1,20))
    elements.append(Paragraph(f"Patient ID: {patient_id}", styles['Normal']))
    elements.append(Paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    elements.append(Spacer(1,20))
    
    # Prediction
    elements.append(Paragraph("Model Prediction", styles['Heading2']))
    elements.append(Paragraph(
        f"Prediction: {'Treatment Likely Successful' if result['prediction']==1 else 'Treatment Likely Unsuccessful'}",
        styles['Normal']
    ))
    elements.append(Paragraph(f"Probability of response: {result['probability']:.2%}", styles['Normal']))
    
    # SHAP
    if shap_df is not None:
        elements.append(Paragraph("Feature Contributions (SHAP)", styles['Heading2']))
        elements.append(Spacer(1,10))
        shap_img = create_shap_plot(shap_df)
        elements.append(Image(shap_img, width=400, height=250))
        elements.append(Spacer(1,20))
    
    # Explanation
    if explanation_text:
        elements.append(Paragraph("Model Explanation", styles['Heading2']))
        elements.append(Paragraph(explanation_text, styles['Normal']))
        elements.append(Spacer(1,20))

    # Disclaimer
    disclaimer_text = (
        "Disclaimer: This report is for informational purposes only. "
        "It does not constitute medical advice and should not be used as the sole basis for clinical decisions. "
        "Clinicians should use their professional judgment and consult relevant guidelines and specialists before making treatment decisions."
    )
    elements.append(Paragraph("Disclaimer", styles['Heading2']))
    elements.append(Paragraph(disclaimer_text, styles['Normal']))
    elements.append(Spacer(1, 20))

    doc.build(elements)
    buffer.seek(0)
    return buffer

def show_pdf(pdf_file):
    pdf_bytes = pdf_file.getvalue()
    b64 = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)