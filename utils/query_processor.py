import os
import streamlit as st
from groq import Groq
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
    
    # Build conversation history string
    conv_history_str = ""
    for msg in conversation_history[-4:]:  # Limit to last 4 messages for context
        if msg["role"] == "user":
            conv_history_str += f"User: {msg['content']}\n"
        else:
            conv_history_str += f"Assistant: {msg['content']}\n"
    
    # Get system prompt and few-shot examples
    system_prompt = get_system_prompt()
    few_shot_examples = get_few_shot_examples()
    
    # Craft the prompt in Llama 3 format with few-shot examples
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>"
    
    # Add few-shot examples in the conversation history
    prompt += f"<|start_header_id|>user<|end_header_id|>\nHere are some examples of how you should respond:\n{few_shot_examples}<|eot_id|>"
    prompt += f"<|start_header_id|>assistant<|end_header_id|>\nI understand. I'll follow these examples when answering questions about UCONN courses.<|eot_id|>"
    
    # Add recent conversation history
    if conv_history_str:
        prompt += f"<|start_header_id|>user<|end_header_id|>\nHere's our recent conversation history:\n{conv_history_str}<|eot_id|>"
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\nThank you for providing the conversation history. I'll keep it in mind for context.<|eot_id|>"
    
    # Add context from retrieved documents
    prompt += f"<|start_header_id|>user<|end_header_id|>\nHere is relevant information from the UCONN course catalog:\n{context}<|eot_id|>"
    prompt += f"<|start_header_id|>assistant<|end_header_id|>\nThank you for providing the context. I'll use this information to answer accurately.<|eot_id|>"
    
    # Add the actual user query
    prompt += f"<|start_header_id|>user<|end_header_id|>\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
    # Call Groq API
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.2,  # Lower temperature for more factual responses
        max_tokens=1024,
    )
    
    response = chat_completion.choices[0].message.content
    
    # Identify potential source URLs and add to response if not already included
    source_urls = identify_source_urls(context)
    if source_urls and "https://" not in response:
        response += "\n\nSources:"
        for name, url in source_urls:
            response += f"\n- {name}: {url}"
    
    return response