import streamlit as st
import chromadb
import os
import time
from groq import Groq
from dotenv import load_dotenv
from tools import generate_rag_response

# 1. Page Configuration
st.set_page_config(
    page_title="NLP Expert Assistant",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom Styling (Clean & Structured White Theme)
st.markdown("""
<style>
    /* Main Background - Crisp White Setup */
    .stApp {
        background-color: #ffffff;
        color: #1e1e1e;
    }
    
    /* Sidebar styling - Slightly off-white for contrast */
    [data-testid="stSidebar"] {
        background-color: #f7f9fc;
        border-right: 1px solid #e0e4e8;
    }
    
    /* Make the top padding consistent */
    .main {
        padding-top: 1rem;
    }

    /* Style Chat Containers - Clean Outline */
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #e5e9f0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04);
        color: #2b2b2b;
    }

    /* Input area styling */
    .stChatInputContainer {
        padding-bottom: 2rem;
    }
    
    /* Divider Customization */
    hr {
        border-top: 1px solid #e0e4e8;
    }
    
    /* Headers Customization */
    h1, h2, h3 {
        color: #111111;
        text-shadow: none;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

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
    st.markdown("**Search:** Vector + Context Synthesis")
    
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
    client = chromadb.PersistentClient(path="vectordb/chroma_db")
    return client.get_or_create_collection("NLP_Book"), Groq(api_key=os.getenv("GROQ_API_KEY"))

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
if prompt := st.chat_input("E.g., What is the difference between Generative and Discriminative models?"):
    # Duplicate check
    past_queries = [m["content"] for m in st.session_state.msgs if m["role"] == "user"]
    
    st.session_state.msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"): 
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="🤖"):
        if prompt in past_queries:
            st.info("That question was already answered! Please check the history above.")
            response = "That question was already answered! Please check the history above."
            st.session_state.msgs.append({"role": "assistant", "content": response})
        else:
            with st.spinner("Searching the text and synthesizing the answer..."):
                start_time = time.time()
                try:
                    response = generate_rag_response(prompt, col, groq)
                    generation_time = time.time() - start_time
                    st.markdown(response)
                    st.caption(f"⏱️ Response generated in {generation_time:.2f} seconds")
                    st.session_state.msgs.append({"role": "assistant", "content": response})
                except Exception as e:
                    response = f"An error occurred while generating the response: {e}"
                    st.error(response)
                    st.session_state.msgs.append({"role": "assistant", "content": response})