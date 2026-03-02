from router import router 
from tools import weather
from tools import web_search
from tools import vector_db

def orchestrate_query(query: str):
    route = router(query)
    if route == "weather":
        print("weather is called")
        context = weather()
    elif route == "web_search":
        print("web search is called")
        context = web_search()
    elif route == "vector_db":
        print("vector db is called")
        context = vector_db(query)

    
    
    return context

print(orchestrate_query("I wanna know about the weather of Lahore?"))