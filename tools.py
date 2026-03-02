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


    print(call_groq(query,prompt))





