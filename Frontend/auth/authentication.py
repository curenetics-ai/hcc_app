import streamlit as st 
import os 
from dotenv import load_dotenv

load_dotenv()

PASSWORD = os.getenv("APP_PASSWORD")

def Login():

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:

        st.title("Secure Access")

        password = st.text_input("Enter password", type="password")

        if st.button("Login"):
            if password == PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")

        st.stop()