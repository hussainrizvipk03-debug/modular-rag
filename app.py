import streamlit as st
import sys
import os
import time

# 🛠️ Deployment Patch: Ensure SQLite compatibility on Streamlit Cloud
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from dotenv import load_dotenv
from groq import Groq
from tools import generate_rag_response

load_dotenv()

# 1. Page Configuration
st.set_page_config(
    page_title="NLP Expert Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom Styling (Clean & Structured White Theme)
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
        color: #1e1e1e;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f7f9fc;
        border-right: 1px solid #e0e4e8;
    }
    
    .main {
        padding-top: 1rem;
    }

    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #e5e9f0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04);
        color: #2b2b2b;
    }

    .stChatInputContainer {
        padding-bottom: 2rem;
    }
    
    hr {
        border-top: 1px solid #e0e4e8;
    }
    
    h1, h2, h3 {
        color: #111111;
        text-shadow: none;
    }
</style>
""", unsafe_allow_html=True)


# 🔐 Secure API Key Handling for Streamlit Cloud & Local
GROK_API_KEY = os.getenv("GROK_API_KEY")

if not GROK_API_KEY and "GROK_API_KEY" in st.secrets:
    GROK_API_KEY = st.secrets["GROK_API_KEY"]

if not GROK_API_KEY:
    st.error("Missing API key! Please set GROK_API_KEY in Streamlit secrets or your local .env file.")
    st.stop()


# 3. Sidebar for Info/Settings
with st.sidebar:
    st.title("📚 NLP Assistant")
    st.markdown("### About")
    st.markdown(
        "This is a **Modular RAG** (Retrieval-Augmented Generation) assistant "
        "specializing in Natural Language Processing concepts. "
        "It uses a local vector database to retrieve precise context from your textbook."
    )
    st.divider()
    st.markdown("**Model:** `llama-3.3-70b-versatile`")
    st.markdown("**Engine:** ChromaDB + Groq")
    
    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.msgs = [
            {"role": "assistant", "content": "Hello! I am your NLP Expert Assistant. How can I help you understand the concepts today?"}
        ]
        st.rerun()


# 4. Main Chat Interface
st.title("🤖 Chat with your NLP Chatbot")
st.markdown("Ask complex conceptual questions, ask for mathematical formulas, or compare NLP techniques.")


@st.cache_resource(show_spinner="Initializing Database and LLM...")
def init():
    try:
        chroma_client = chromadb.PersistentClient(path="vectordb/chroma_db")
        collection = chroma_client.get_or_create_collection("NLP_Book")
        groq_client = Groq(api_key=GROK_API_KEY)
        return collection, groq_client
    except Exception as e:
        raise RuntimeError(f"Initialization failed: {e}")


try:
    col, groq = init()
except Exception as e:
    st.error(f"Error initializing services: {e}")
    st.stop()


# 5. Chat History Management
if "msgs" not in st.session_state:
    st.session_state.msgs = [
        {"role": "assistant", "content": "Hello! I am your NLP Expert Assistant. How can I help you understand the concepts today?"}
    ]


# Display chat messages
for msg in st.session_state.msgs:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# 6. User Input and Response Generation
if prompt := st.chat_input():
    
    # Store history BEFORE appending current prompt to check for duplicates
    past_queries = [m["content"] for m in st.session_state.msgs if m["role"] == "user"]
    
    st.session_state.msgs.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="🤖"):
        
        if prompt in past_queries:
            # Find the previous assistant response to this prompt
            previous_answer = ""
            for i in range(len(st.session_state.msgs) - 2, -1, -1):
                if st.session_state.msgs[i]["role"] == "user" and st.session_state.msgs[i]["content"] == prompt:
                    if i + 1 < len(st.session_state.msgs) and st.session_state.msgs[i+1]["role"] == "assistant":
                        previous_answer = st.session_state.msgs[i+1]["content"]
                        break
            
            not_found_msg = "this is not in the provided context."
            if not_found_msg in previous_answer.lower():
                response = not_found_msg
                st.error(response)
            else:
                # Show the question and the previous "detail" (the answer)
                response = f"**Already Asked:** {prompt}\n\n**Answer:**\n{previous_answer}"
                st.warning(response)
                
            st.session_state.msgs.append({"role": "assistant", "content": response})
        
        else:
            with st.spinner("Searching the text and synthesizing the answer..."):
                start_time = time.time()
                try:
                    # Pass the message history for conversational context
                    response = generate_rag_response(prompt, col, groq, st.session_state.msgs)
                    generation_time = time.time() - start_time
                    
                    st.markdown(response)
                    st.caption(f"⏱️ Response generated in {generation_time:.2f} seconds")
                    
                    st.session_state.msgs.append({"role": "assistant", "content": response})
                
                except Exception as e:
                    error_msg = f"An error occurred: {e}"
                    st.error(error_msg)
                    st.session_state.msgs.append({"role": "assistant", "content": error_msg})