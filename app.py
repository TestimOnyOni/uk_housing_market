import streamlit as st
import pandas as pd
import pickle
from rag_core import KBContext, XGBPredictor, handle_query

# Load models and artifacts
@st.cache_resource
def load_kb_and_model():
    # Load KB (index + encoders + stats)
    kb = KBContext.load("artifacts")

    # Load trained predictor
    with open("artifacts/model_xgb.pkl", "rb") as f:
        predictor = pickle.load(f)

    return kb, predictor


# Streamlit UI
st.set_page_config(page_title="UK Housing RAG + XGBoost", layout="centered")

st.title("🏠 Property Price Estimator & Knowledge Base")

st.markdown("""
This app combines **retrieval-augmented generation (RAG)** with an **XGBoost price predictor**.

* Ask a question in plain English, like:

  * *Estimate price for a 4-bed townhouse in Shoreditch built 2008*
  * *Show comparable listings in Guildford with 3 bedrooms*
    """)

# Load resources
kb, predictor = load_kb_and_model()

# User input
query = st.text_area("Enter your property-related query:", height=100)

if st.button("Submit") and query.strip():
    with st.spinner("Processing..."):
        try:
            answer = handle_query(query, kb, predictor, llm=None)
            st.markdown("### 📌 Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Error: {e}")

st.sidebar.header("About")
st.sidebar.info("Prototype: RAG + XGBoost Housing Valuation App.")
