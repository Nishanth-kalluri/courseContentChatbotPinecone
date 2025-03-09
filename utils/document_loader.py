import requests
import fitz  # PyMuPDF for PDFs
import streamlit as st
from bs4 import BeautifulSoup
from data.urls import pdf_urls, undergraduate_courses_url, graduate_courses_url
from data.course_codes import undergraduate_codes, graduate_codes

def get_pdf_urls():
    """Return list of PDF URLs"""
    return pdf_urls

def get_course_urls():
    """Construct and return list of course URLs"""
    dynamic_urls = []
    for code in undergraduate_codes:
        url = f"{undergraduate_courses_url}{code}/"
        dynamic_urls.append(url)
    for code in graduate_codes:
        url = f"{graduate_courses_url}{code}/"
        dynamic_urls.append(url)
    return dynamic_urls

def download_pdf(url, status):
    """Download and extract text from PDF"""
    status.update(label=f"Downloading PDF: {url}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            pdf_bytes = response.content
            # Open PDF from bytes using PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text("text") + "\n\n"
            status.update(label=f"✅ Processed PDF: {url}")
            return text
        else:
            status.update(label=f"❌ Failed to download PDF: {url}")
            return None
    except Exception as e:
        status.update(label=f"Error processing PDF {url}: {e}")
        return None

def download_webpage(url, status):
    """Download and extract text from webpage"""
    status.update(label=f"Downloading webpage: {url}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            html_content = response.text
            soup = BeautifulSoup(html_content, "html.parser")
            # Extract visible text from the webpage
            text = soup.get_text(separator="\n")
            status.update(label=f"✅ Processed webpage: {url}")
            return text
        else:
            status.update(label=f"❌ Failed to download webpage: {url}")
            return None
    except Exception as e:
        status.update(label=f"Error processing webpage {url}: {e}")
        return None