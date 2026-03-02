from llm_client import call_groq

def router(query):

    system_prompt = """
    You are an expert query router. Your task is to categorize the user's query into one of the following categories:
    1. 'web_search': For general queries, latest news, or information search the web for that.
    2. 'vector_db': For queries requiring specific document knowledge or technical data search vector db
    3. 'weather': For any query regarding current weather or forecasts.
    
    Respond ONLY with the category name (web_search, vector_db, or weather).
    """

    return call_groq(query, system_prompt)
