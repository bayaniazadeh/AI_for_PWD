import pandas as pd
import requests
import numpy as np
import pandas as pd
import ollama
import pandas as pd
from json import loads, dumps


# Disability categories:
def Screening_v7(text):
    try:
        # Assuming ollama.pull is necessary before chat (based on your original code)
        ollama.pull('llama3.2')
        
        # Chat with the model
        response = ollama.chat(
            model='llama3.2:latest',
            messages=[
                {
                    'role': 'user',
                    'content': text,
                },
                {
                    'role': 'user',
                    'content': (
                        f"You are an expert in screening full texts for scoping reviews. "
                        f"Given the following full text:\n{text}\n"
                        f"Determine whether to **Include** or **Exclude** it for a full-text review based on the following strict criteria:\n\n"
                        f"**Include** only if:\n"
                        f"- The study focuses on people with disabilities include physical, mental, or developmental impairments (e.g., autism, mobility impairments, cognitive challenges).\n"
                        f"- The study uses Artificial Intelligece (AI ) or Machine Learning (ML) models including predictive models, language models, diagnostic tools, diagnostic models, interventions, or assessments(e.g., supervised, unsupervised, deep learning, Generative Adversarial Networks, Bayesian networks).\n"
                        f"- The study targets disability outcomes or related metrics (e.g., psychomotor skills, physical health, progression risk).\n"
                        f"- The study focuses on general disease prediction algorithms for any type of disability, such as auditory, vision, or mobility problems, or intellectual disabilities such as dementia, Alzheimers, or autism.\n"
                        f"- The paper focuses on Brain-Computer Interface for stroke neurorehabilitation with AI.\n\n"
                        f"**Exclude** if:\n"
                        f"- The study does not involve AI/ML, or the AI/ML models are unrelated to disabilities.\n"
                        f"- The focus is solely on assistive technologies without AI/ML (e.g., wearables, Bluetooth).\n"
                        f"- The study type includes review, meta-analysis, book chapter, conference paper, poster, feasibility, or pilot studies without reporting AI/ML performance.\n"
                        f"- The primary focus is on general disease prediction algorithms with no disability-specific context.\n\n"
                        f"**Additional Guidance**:\n"
                        f"- If uncertain or the connection to disabilities, always default to **Exclude** and explain your reasoning.\n\n"
                        f"Examples:\n"
                        f"- Include: A Bayesian network for assessing physical fitness in students with developmental disabilities.\n"
                        f"- Exclude: A Bluetooth-based or Fuzzy assistive device that is not AI/ML.\n\n"
                        f"Respond strictly with one word, '**Include**' or '**Exclude**', followed by a brief explanation of your choice."
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

