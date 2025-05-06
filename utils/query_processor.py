import os
import streamlit as st
from groq import Groq
import re
from utils.document_processor import identify_source_urls
from data.prompts import get_system_prompt, get_few_shot_examples

def process_query(query, conversation_history):
    """Process user queries with few-shot prompting"""
    # Retrieve relevant documents
    relevant_docs = st.session_state.vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 5}
    ).get_relevant_documents(query)
    
    context = "\n".join([doc.page_content for doc in relevant_docs])
    source_urls = identify_source_urls(context)
    
    # Get system prompt and few-shot examples
    system_prompt = get_system_prompt()
    few_shot_examples = get_few_shot_examples()

    # Build messages list for DeepSeek format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here are response examples to follow:\n{few_shot_examples}"},
        {"role": "assistant", "content": "Understood. I will follow the examples and catalog structure when responding."}
    ]
    
    # Add conversation context (last 3 exchanges)
    for msg in conversation_history[-6:]:  # Keep last 3 pairs (6 messages)
        messages.append(msg)
    
    # Add retrieved context with source tracking
    context_message = f"Current course catalog context:\n{context}"
    if source_urls:
        context_message += "\n\nAvailable sources:\n" + "\n".join([f"- {url}" for _, url in source_urls])
    messages.extend([
        {"role": "user", "content": context_message},
        {"role": "assistant", "content": "Context received. Will use verified information from provided sources."}
    ])
    
    # Add current query
    messages.append({"role": "user", "content": query})
    
    # Call Groq API with proper message format
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="deepseek-r1-distill-llama-70b",
        temperature=0.3,
        max_tokens=2048,
        stop=["<|eot_id|>"],  # Add stop sequence if needed
        top_p=0.9,
    )
    
    # Get raw response from model
    raw_response = chat_completion.choices[0].message.content
    
    # Clean the response to remove thinking blocks
    cleaned_response = clean_response(raw_response)
    
    # Add source citations if missing
    if source_urls and not any(url in cleaned_response for _, url in source_urls):
        cleaned_response += "\n\nSources:\n" + "\n".join([f"- {url}" for _, url in source_urls])
    
    return cleaned_response


def clean_response(response):
    """
    Removes thinking blocks (<think>...</think>) from the model response.
    
    Args:
        response (str): The raw model response
        
    Returns:
        str: The cleaned response with thinking blocks removed
    """
    # Check if thinking block exists
    if '<think>' in response and '</think>' in response:
        # Use regex to remove everything between <think> and </think> tags, including the tags
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Remove any leading whitespace that might remain after removing the thinking block
        cleaned = cleaned.lstrip()
        
        return cleaned
    else:
        # If no thinking block is present, return the original response
        return response