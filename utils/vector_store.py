import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_INDEX_NAME, PINECONE_NAMESPACE, EMBEDDING_DIMENSION
from utils.document_processor import process_documents

def initialize_pinecone():
    """Initialize Pinecone vector store"""
    if st.session_state.pinecone_initialized:
        return True
        
    # Initialize Pinecone
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Check if index exists
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        try:
            # Create index if it doesn't exist
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,  # Dimension for Google's embedding model
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            st.write("Pinecone index created successfully.")
        except Exception as e:
            st.error(f"Error creating Pinecone index: {e}")
            return False
    
    st.session_state.pinecone_initialized = True
    return True

def load_data():
    """Load and process data with Pinecone integration"""
    # Initialize Pinecone
    if not initialize_pinecone():
        return None
        
    # Create embeddings instance
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Check if vectors already exist in Pinecone
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    try:
        stats = index.describe_index_stats()
        vector_count = stats.namespaces.get(PINECONE_NAMESPACE, {}).get("vector_count", 0)
        
        if vector_count > 0:
            # Vectors already exist in Pinecone - retrieve them
            st.write(f"Found {vector_count} existing vectors in Pinecone. Loading...")
            vectorstore = PineconeVectorStore(
                index_name=PINECONE_INDEX_NAME,
                embedding=embeddings,
                namespace=PINECONE_NAMESPACE
            )
            return vectorstore
    except Exception as e:
        st.write(f"Error checking Pinecone stats: {e}")
        # Continue with data processing if there's an error
    
    # If we get here, we need to process the data and create embeddings
    with st.status("Processing documents and creating embeddings...", expanded=True) as status:
        # Process documents
        chunks, _ = process_documents(status)
        
        if not chunks:
            return None
        
        # Create embeddings and build a vector store using Pinecone
        status.update(label="Creating embeddings and upserting to Pinecone...")
        vectorstore = PineconeVectorStore.from_texts(
            chunks,
            embeddings,
            index_name=PINECONE_INDEX_NAME,
            namespace=PINECONE_NAMESPACE
        )
        status.update(label="Embeddings successfully created and stored in Pinecone!", state="complete")
        
        return vectorstore