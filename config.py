import os
import streamlit as st
from dotenv import load_dotenv

#use this setup function if using .env
# def setup_environment():
#     """Set up environment variables from .env file"""
#     load_dotenv()
    
#     # Check if API keys are set in session state
#     if "api_keys_set" not in st.session_state:
#         # API keys should be loaded from .env file
#         os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
#         os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
#         os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY", "")
#         st.session_state.api_keys_set = True

def setup_environment():
    """Set up environment variables from Streamlit secrets"""
    # Check if API keys are already set in session state
    if "api_keys_set" not in st.session_state:
        # Load API keys from st.secrets
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
        st.session_state.api_keys_set = True


def check_environment_variables():
    """Check if all required environment variables are set"""
    required_vars = ["GOOGLE_API_KEY", "GROQ_API_KEY", "PINECONE_API_KEY"]
    for var in required_vars:
        if not os.getenv(var):
            return False
    return True

# Constants for Pinecone configuration
PINECONE_INDEX_NAME = "uconn-course-catalog"
PINECONE_NAMESPACE = "course_catalog"
EMBEDDING_DIMENSION = 768  # Dimension for Google's embedding model