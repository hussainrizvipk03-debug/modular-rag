import chromadb
from groq import Groq
from llm_client import call_groq



def weather():
    print("it is a sunny day")

def web_search():
    print("virat kohli is the greatest of all time")

def vector_db(query, top_k=10):
    client = chromadb.PersistentClient(path="vectordb/chroma_db")
    col = client.get_collection(name="NLP_Book")
    results = col.query(query_texts=[query], n_results=top_k)
    retrieved_docs = results["documents"][0]
    return "\n\n".join(retrieved_docs)

def generate_rag_response(user_query, col, client, chat_history=None):
    # 1. Retrieve Context with higher top_k for better synthesis
    results = col.query(query_texts=[user_query], n_results=10)
    retrieved_docs = results["documents"][0] if results["documents"] else []
    context = "\n\n".join(retrieved_docs)
    
    if not context.strip():
        return "I'm sorry, I couldn't find relevant information in the document to answer that."

    # 2. Build Conversational Prompt with Extreme Grounding
    strict_system_prompt = (
        "You are a strict and factual NLP Expert Assistant. Your ONLY source of truth is the provided Context.\n"
        "CRITICAL RULES:\n"
        "1. Answer ONLY from the Context. If the answer is not explicitly there, say: 'this is not in the provided context.'\n"
        "2. Do NOT use outside knowledge, even for definitions or common facts not listed in the context.\n"
        "3. Do NOT provide 'related info' or guess. If only 50% of the answer is in the context, only provide that 50%.\n"
        "4. Use the Chat History solely to understand what 'it', 'they', or 'that' refers to in follow-up questions."
    )

    # Prepare messages including history
    messages = [{"role": "system", "content": strict_system_prompt}]
    
    if chat_history:
        # Filter history to only include standard user/assistant messages
        # (Exclude system notes or duplicate warnings to keep context clean)
        for msg in chat_history[-4:]:
            if "Already asked" not in msg["content"] and "already asked" not in msg["content"]:
                messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add context to the current query
    messages.append({
        "role": "user", 
        "content": f"REFER TO THIS CONTEXT FOR YOUR RESPONSE:\n{context}\n\nQUESTION:\n{user_query}"
    })

    # 3. Generate Response
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=900,
        temperature=0, 
        messages=messages
    )
    
    return response.choices[0].message.content
