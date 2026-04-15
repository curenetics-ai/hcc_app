import streamlit as st 

def init_session_state():

    defaults = {
        'results': None,
        'has_predictions': False,
        'explanation_text': None,
        'explanation requested': False,
    }

    for key,value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value