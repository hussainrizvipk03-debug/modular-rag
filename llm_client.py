from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()
def call_groq(query, system_prompt):
    client = Groq(api_key=os.getenv("groq_api_key"))
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=200,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content


    
