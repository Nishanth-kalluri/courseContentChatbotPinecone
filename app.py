import streamlit as st
import os
import time
from utils.document_loader import get_pdf_urls, get_course_urls
from utils.document_processor import extract_doc_ids
from utils.vector_store import initialize_pinecone, load_data
from utils.query_processor import process_query
from config import setup_environment, check_environment_variables

# Page config for better appearance
st.set_page_config(page_title="Course Catalog Chatbot", layout="wide")

# Set up environment variables
setup_environment()

# Initialize session state variables
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "source_urls" not in st.session_state:
    st.session_state.source_urls = {}
if "pinecone_initialized" not in st.session_state:
    st.session_state.pinecone_initialized = False

# Title and description
st.title("UCONN Course Advisor")
st.write("""
This AI advisor helps you navigate course catalogs and choose the right courses based on your academic 
and career goals. It provides personalized recommendations and information about UCONN courses, programs, 
prerequisites, and more.
""")

# Create a container for the chat interface
chat_container = st.container()

# Sidebar with data loading button (minimal UI)
with st.sidebar:
    st.header("Controls")
    
    # Check if environment variables are set
    env_status = check_environment_variables()
    
    # Data loading section
    st.subheader("Data Management")
    load_disabled = not env_status
    
    if st.button("Load Course Catalog Data", disabled=load_disabled):
        with st.spinner("Loading data... This may take several minutes if embeddings need to be created"):
            st.session_state.vectorstore = load_data()
            st.session_state.data_loaded = True
            st.success("✅ Data loaded successfully!")
    
    # Show data status
    if st.session_state.data_loaded:
        st.success("Data is loaded and ready for queries")
    elif not env_status:
        st.error("API keys not found. Please set up your .env file.")
    else:
        st.warning("Please load data before chatting")
    
    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared!")

# Main chat interface with proper positioning
with chat_container:
    # Create a container for chat messages that takes up most of the screen
    chat_container = st.container()

# Display chat messages first
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input at the bottom of the page
prompt = st.chat_input("Ask about UCONN courses and programs...")
if prompt:
    if not st.session_state.data_loaded:
        st.error("Please load the data first by clicking the 'Load Course Catalog Data' button in the sidebar.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                response = process_query(prompt, st.session_state.messages[:-1])  # Exclude current message
                elapsed_time = time.time() - start_time
            
            st.markdown(response)
            st.caption(f"Response time: {elapsed_time:.2f} seconds")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})