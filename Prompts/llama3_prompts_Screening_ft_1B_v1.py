import pandas as pd
import requests
import numpy as np
import pandas as pd
import ollama
import pandas as pd
from json import loads, dumps


# Disability categories:
def extract_Title_abs(text):
    try:
        # Assuming ollama.pull is necessary before chat (based on your original code)
        ollama.pull('llama3.2:3b')
        
        # Chat with the model
        response = ollama.chat(
            model='llama3.2:3b',
            messages=[
                {
                    'role': 'user',
                    'content': text,
                },
                {
                    'role': 'user',
                    'content': (
                        f"In the following full-text {text}of paper givef me the title and the abstract of text which come in the first page of text"
                    ),
                },
            ]
        )
        
        # Return the content of the response
        return response['message']['content']
    
    except requests.exceptions.ConnectionError:
        return "Connection refused"
    
    except Exception as e:
        return f"Error: {e}"

