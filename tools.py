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

   INSTRUCTIONS:
You are an NLP Assistant. Your goal is to synthesize and extract from the provided CONTEXT to answer the QUERY with both academic rigor and conceptual clarity.

Research Phase: Scrutinize the context for technical specifics, including mathematical formulas (in LaTeX), algorithm steps, time complexity, and specific citations mentioned by Eisenstein.

Explainer Phase: Translate those technicalities into a clear, structured explanation. Define all specialized terminology used in your answer.

Comprehensive Synthesis: Do not ignore any part of the query. If the query asks for "differences," "operations," or "comparisons," ensure you provide a structured list or a contrasted analysis.

Identity Awareness: Treat the author (Eienstein) as the primary source. If a concept is described as a "standard approach" or "proposed solution" in the text, attribute it to the textbook's framework.

Quality Control: If the specific answer (like a list of operations) is missing from the context, state exactly what is missing but explain the closest related concept found.
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
        max_tokens=900,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant. Answer the user's question concisely using ONLY the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_query}"}
        ]
    )
    
    return response.choices[0].message.content
