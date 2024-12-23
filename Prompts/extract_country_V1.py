import pandas as pd
import requests
import ollama
from json import loads, dumps

# Initialize the model once
ollama.pull('llama3.2:3b')

# Function to extract title and abstract
def extract_Country_v2(text):
    try:
        # Chat with the model
        response = ollama.chat(
            model='llama3.2:3b',
            messages=[
                {
                    'role': 'user',
                    'content': (
                        f"From the following full text of the paper: {text}, extract the country that the study has been carried out, only state the name of the country if it is not mentioned state: not adressed without any supplementary explanations."
                    ),
                },
            ]
        )

        # Validate and return the content
        if 'message' in response and 'content' in response['message']:
            return response['message']['content']
        else:
            return "Unexpected response structure from the model."

    except requests.exceptions.ConnectionError:
        return "Connection refused: Unable to connect to the server."

    except Exception as e:
        return f"Error: {e}"
