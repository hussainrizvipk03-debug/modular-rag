from groq import Groq
from dotenv import load_dotenv
import os

def get_api_key():
    load_dotenv()
    api_key = os.environ.get("GROK_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROK_API_KEY")
        except Exception:
            pass
    return api_key

def call_groq(query, system_prompt):
    api_key = get_api_key()
    if not api_key:
        raise ValueError("API Key not found!")
        
    client = Groq(api_key=api_key)
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=200,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content
