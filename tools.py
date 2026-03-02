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
    You have to reply accordingly with the CONTEXT and the QUERY
    """

def is_query_change(query, client):
    """Detects intent and rewrites as a standalone search query."""
    return client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": f"Rewrite this as a standalone search query: {query}"}]
    ).choices[0].message.content

def updated_query(query, client):
    """Optimizes the query for retrieval."""
    return client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": f"Optimize for retrieval: {query}"}]
    ).choices[0].message.content


def confidence_score_part(query, docs, client):
    """Analyzes chunks for relevance and returns score and context."""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Analyze snippets for relevance. Return output as: 'Score: [0-10]\nContext: [snippet text]'"},
            {"role": "user", "content": f"Query: {query}\nSnippets:\n" + "\n---\n".join(docs)}
        ]
    ).choices[0].message.content
    return res

def generate_rag_response(user_query, col, client):
    
    refined = updated_query(is_query_change(user_query, client), client)
    
    
    docs = col.query(query_texts=[refined], n_results=5)['documents'][0]
    

    scored_output = confidence_score_part(user_query, docs, client)
    
    
    score = 0
    try:
        score_line = [l for l in scored_output.split('\n') if 'Score:' in l][0]
        score = int(''.join(c for c in score_line if c.isdigit()))
    except: pass

    
    if score < 5:
        return f"I'm sorry, I couldn't find relevant information in the document to answer that. (Confidence Score: {score}/10)"

    
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"Answer concisely using ONLY the provided context. ALWAYS show the confidence score: {score}/10 at the end."},
            {"role": "user", "content": f"Context: {scored_output}\n\nQuestion: {user_query}"}
        ]
    ).choices[0].message.content




    print(call_groq(is_query_change,updated_query,confidence_score_part))





