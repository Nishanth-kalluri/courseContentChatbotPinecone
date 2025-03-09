import re
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.document_loader import download_pdf, download_webpage, get_pdf_urls, get_course_urls

def extract_doc_ids(text, url):
    """Extract document IDs for tracking source URLs"""
    # Extract course codes or unique identifiers from the text
    # This is a simplified version - in practice, you'd need a more robust pattern
    course_codes = re.findall(r'\b[A-Z]{2,4}\s\d{4}[A-Z]?\b', text)
    doc_id_to_url = {}
    for course_code in course_codes:
        doc_id_to_url[course_code] = url
    return doc_id_to_url

def process_documents(status):
    """Process all documents and return text chunks and source mapping"""
    all_texts = []
    doc_sources = {}

    # Process PDF URLs
    pdf_urls = get_pdf_urls()
    for url in pdf_urls:
        text = download_pdf(url, status)
        if text:
            all_texts.append(text)
            doc_sources.update(extract_doc_ids(text, url))

    # Process dynamic webpage URLs constructed from course codes
    dynamic_urls = get_course_urls()
    for url in dynamic_urls:
        text = download_webpage(url, status)
        if text:
            all_texts.append(text)
            doc_sources.update(extract_doc_ids(text, url))

    # Save source mapping in session state
    st.session_state.source_urls = doc_sources

    if not all_texts:
        status.update(label="No text was extracted from the provided sources.", state="error")
        return None, None

    # Combine all extracted texts
    combined_text = "\n\n".join(all_texts)
    status.update(label=f"Extracted {len(combined_text)} characters of text from all sources.")

    # Split the text into chunks
    status.update(label="Splitting text into manageable chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(combined_text)
    status.update(label=f"Text split into {len(chunks)} chunks.")

    return chunks, doc_sources

def identify_source_urls(content):
    """Identify potential source URLs based on content"""
    urls = []
    for code, url in st.session_state.source_urls.items():
        if code in content:
            urls.append((code, url))
    
    # If no specific course codes found, return most relevant URLs
    if not urls:
        pdf_urls = get_pdf_urls()
        if "undergraduate" in content.lower():
            urls.append(("Undergraduate Catalog", pdf_urls[0]))
        if "graduate" in content.lower():
            urls.append(("Graduate Catalog", pdf_urls[1]))
    
    return list(set(urls))[:3]  # Return up to 3 unique sources