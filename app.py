import streamlit as st, chromadb, os
from groq import Groq
from dotenv import load_dotenv
from tools import generate_rag_response

load_dotenv()
st.set_page_config(page_title="PDF AI", page_icon="📄")
st.title(" NLP Chat Assistant")

@st.cache_resource
def init():
    client = chromadb.PersistentClient(path="vectordb/chroma_db")
    return client.get_or_create_collection("NLP_Book"), Groq(api_key=os.getenv("GROQ_API_KEY"))

col, groq = init()

if "msgs" not in st.session_state: st.session_state.msgs = []

for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input("Ask about your PDF"):
    # Duplicate check
    past_queries = [m["content"] for m in st.session_state.msgs if m["role"] == "user"]
    
    with st.chat_message("user"): st.write(prompt)
    st.session_state.msgs.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        if prompt in past_queries:
            response = "That question was already answered! Please check the history above."
        else:
            response = generate_rag_response(prompt,col,groq)
        
        st.write(response)
        st.session_state.msgs.append({"role": "assistant", "content": response})