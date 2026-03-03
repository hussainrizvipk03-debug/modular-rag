import chromadb
from groq import Groq
from llm_client import call_groq



def weather():
    print("it is a sunny day")

def web_search():
    print("virat kohli is the greatest of all time")

def vector_db(query,top_k=3):
    client = chromadb.PersistentClient(path="vectordb/chroma_db")

    col = client.get_collection(
        name="NLP_Book",
    )

    results = col.query(
        query_texts=[query],
        n_results=top_k
    )

    retrieved_docs = results["documents"][0]

    context = "\n\n".join(retrieved_docs)



    prompt = f"""
    CONTEXT:
    {context}

    QUERY:
    {query}

    You are an NLP Assistant . 
    You have been provided with the CONTEXT and the QUERY . 
    You have to reply accordingly with the CONTEXT and the QUERY .
    """

    return context

def generate_rag_response(user_query, col, client):
    # Directly query the VectorDB for context
    results = col.query(query_texts=[user_query], n_results=3)
    
    # Extract retrieved documents
    retrieved_docs = results["documents"][0] if results["documents"] else []
    context = "\n\n".join(retrieved_docs)
    
    if not context.strip():
        return "I'm sorry, I couldn't find relevant information in the document to answer that."

    # Directly retrieve output info using Groq model
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant. Answer the user's question concisely using ONLY the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_query}"}
        ]
    )
    
    return response.choices[0].message.content
