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
    # 1. Retrieve Context with similarity threshold
    results = col.query(query_texts=[user_query], n_results=10)
    
    # Check similarity distance (ChromaDB default is L2; lower is better)
    # If the closest match is too far (e.g., > 1.6), it's likely irrelevant
    distances = results.get("distances", [[]])[0]
    if not distances or distances[0] > 1.6:
        return "this is not in the provided context."

    retrieved_docs = results["documents"][0] if results["documents"] else []
    context = "\n\n".join(retrieved_docs)
    
    if not context.strip():
        return "this is not in the provided context."

    # 2. Build Conversational Prompt with Extreme Grounding
    strict_system_prompt = (
        "You are a strict and factual NLP Expert Assistant. Your ONLY source of truth is the provided Context.\n"
        "CRITICAL RULES:\n"
        "1. If the exact answer is NOT contained in the provided Context, you MUST respond with EXACTLY the following phrase and NOTHING ELSE: 'this is not in the provided context.'\n"
        "2. Do NOT provide apologies, partial answers, or outside knowledge.\n"
        "3. Do NOT guess. If the context is even slightly insufficient, use the 'not in context' phrase.\n"
        "4. Use the Chat History ONLY to resolve pronouns (it, they) to ensure you are looking for the right thing in the Context."
    )

    # Prepare messages including history
    messages = [{"role": "system", "content": strict_system_prompt}]
    
    if chat_history:
        for msg in chat_history[-4:]:
            if "Already asked" not in msg["content"] and "already asked" not in msg["content"]:
                messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({
        "role": "user", 
        "content": f"REFER TO THIS CONTEXT FOR YOUR RESPONSE:\n{context}\n\nQUESTION:\n{user_query}"
    })

    # 3. Generate Response
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=900,
        temperature=0, 
        messages=messages
    )
    
    return response.choices[0].message.content
