import pandas as pd
import requests
import numpy as np
import pandas as pd
import ollama
import pandas as pd
from json import loads, dumps

ollama.pull('llama3.2:3b')

# Disability categories:
def Screening_v4(text):
    try:
        # Assuming ollama.pull is necessary before chat (based on your original code)
       
        
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
                         f"""
You are an expert in screening titles and abstracts for scoping reviews. 
Analyze the following paper, which includes its title and abstract:
{text}

**Inclusion Criteria**:
- Focuses on **Artificial Intelligence (AI)** or **Machine Learning (ML)** applications for **patients with disabilities**.
- Utilizes one or more of the following AI/ML models:
  - **Supervised learning** (e.g., classification, regression).
  - **Unsupervised learning** (e.g., clustering, dimensionality reduction).
  - **Deep learning** (e.g., CNNs, RNNs, transformers).
  - **Natural Language Processing (NLP)** (e.g., named entity recognition, sentiment analysis, text summarization).
- The application specifically addresses disability-related outcomes, treatments, or interventions.
- By disability I mean all types of disabilities such as intellevtual disability, physical impairements and mobility disabilities.

**Exclusion Criteria**:
- Does not focus on AI/ML for patients with disabilities.
- Primarily discusses assistive technologies or mechanical/physical rehabilitation without AI involvement.
- Focuses on feasibility studies or statistical analyses without presenting AI/ML model performance.
- Is a systematic review, scoping review, meta-analysis, conference paper, or poster.

**Your Task**:
Respond with **Include** or **Exclude** and provide a brief justification for your decision.
"""
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

