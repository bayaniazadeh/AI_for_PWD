import pandas as pd
import requests
import ollama
from json import loads, dumps

# Initialize the model once
ollama.pull('llama3.2:3b')

def extract_Title_abs_v3(text):
    try:
        response = ollama.chat(
            model='llama3.2:3b',
            messages=[
                {
                    'role': 'user',
                    'content': (
                        f"From the following full-text: {text}, extract the title and abstract. "
                        f"Return the result in the format 'Title: [TITLE], Abstract: [ABSTRACT]'."
                    ),
                },
            ]
        )

        # Extract the content
        if 'message' in response and 'content' in response['message']:
            content = response['message']['content']

            # Split content into title and abstract
            if "Title:" in content and "Abstract:" in content:
                parts = content.split("Abstract:")
                title = parts[0].replace("Title:", "").strip()
                abstract = parts[1].strip()
                return title, abstract

            return None, None  # If format is unexpected
        else:
            return None, None

    except requests.exceptions.ConnectionError:
        return None, None

    except Exception as e:
        print(f"Error: {e}")
        return None, None
