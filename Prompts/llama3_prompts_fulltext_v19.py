import pandas as pd
import requests
import ollama
from json import loads, dumps

# Disability categories screening function:
def fulltext_screening_v1(text):
    try:
        # Ensure the model is loaded before chat
        ollama.pull('llama3.2')
        
        # Chat with the model
        response = ollama.chat(
            model='llama3.2:latest',
            messages=[
                {
                    'role': 'user',
                    'content': f"""You are an expert in screening full texts for scoping reviews. 
Given the following paper:
{text}
that contains the full text of the paper, determine whether to **Include** or **Exclude** it for a full-text review based on the following strict criteria:

**Include** only if:
- Disabilities include physical, mental, or developmental impairments (e.g., autism, mobility impairments, cognitive challenges).
- The AI/ML models include predictive models, language models, diagnostic tools, diagnostic models, interventions, or assessments (e.g., supervised, unsupervised, deep learning, Generative Adversarial Networks, Bayesian networks).
- The study targets disability outcomes or related metrics (e.g., psychomotor skills, physical health, progression risk).
- The paper focuses on general disease prediction algorithms for any type of disability such as (auditory or vision or mobility problems) or intellectual disabilities such as dementia, alzheimer, autism etc. 
- The paper focuses on Brain Computer Interface for stroke neurorehabilitation with AI.

**Exclude** if:
- The study does not involve AI/ML, or the AI/ML models are unrelated to disabilities.
- The focus is solely on assistive technologies without AI/ML (e.g., wearables, Bluetooth).
- The study type includes reviews, meta-analyses, book chapters, conference papers, posters, feasibility, or pilot studies without reporting AI/ML performance.
- The primary focus is on general disease prediction algorithms with no disability-specific context.

**Additional Guidance**:
- If uncertain or the connection to disabilities is implied, always default to **Exclude** and explain your reasoning.

Examples:
- Include: A Bayesian network for assessing physical fitness in students with developmental disabilities.
- Exclude: A Bluetooth-based or Fuzzy assistive device which is not AI/ML.

Respond strictly with one word, "**Include**" or "**Exclude**", followed by a brief explanation on your choice.
"""


                },
            ]
        )
        
        # Return the content of the response
        return response['message']['content'].strip()
    
    except requests.exceptions.ConnectionError:
        return "Connection refused"
    
    except Exception as e:
        return f"Error: {e}"
