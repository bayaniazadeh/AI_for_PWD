import pandas as pd
import requests
import numpy as np
import pandas as pd
import ollama
import pandas as pd
from json import loads, dumps


# Disability categories:
def Screening_v3(text):
    try:
        # Assuming ollama.pull is necessary before chat (based on your original code)
        ollama.pull('llama3')
        
        # Chat with the model
        response = ollama.chat(
            model='llama3:latest',
            messages=[
                {
                    'role': 'user',
                    'content': text,
                },
                {
                    'role': 'user',
                    'content': f"""You are an expert in Screening the fulltexts of a scoping reviews, given the following
                    {text} 
                    that contains the full text of paper, 
                    Determine whether to **Include** or **Exclude** it for a full-text review based on the following strict criteria:

                    **Include** only if: 
                    - the population of paper consideres people with incapacities and all kinds of disabilities such as (mobility impairment, intellectual disability such as alzheimer, autism etc, hearing loss, vison disability etc.)
                    - *AND* It uses the applications of Artificial Intelligence for the people with disabilities
                    - Disabilities include physical, mental, or developmental impairments (e.g., autism, mobility impairments, cognitive challenges)."
                    - The AI/ML models include predictive models, language models, diagnostic tools, diagnostic models, interventions, or assessments "
                    (e.g., supervised, unsupervised, deep learning, Generative Adversarial Networks, Bayesian networks)."
                    - The study targets disability outcomes or related metrics (e.g., psychomotor skills, physical health, progression risk)."
                    - The paper focuses on general disease prediction algorithms for any type of disability, such as auditory, vision, or mobility problems, "
                    - or intellectual disabilities such as dementia, Alzheimers, or autism."
                    - The paper focuses on Brain-Computer Interface for stroke neurorehabilitation with AI."
                    - The paper calculates the ROC or AUC in the results which is required for evaluating ML models and it focused on its applications for people with disabilities."
                        
                    - It is a research or applied research
                    **Exclude** if: 
                    - It does not consider the people with disabilities
                    - It only considers assistive technologies for movement or mobility functions, which are mechanical technilogies not related to artificial intelligence
                    - It has not use the applications of AI for people with disabilities
                    - It is a systematic review, review, scoping review or meta-analysis
                    - It is a conference paper or poster
                    - It considers diseases prediction algorithms (without considering any disabilities or people in disability situation)
                    - It considers assistive technologies such as devices, systems, or tools that help individuals perform physical tasks without any applications or links to AI
                    Respond strictly with one word, '**Include**' or '**Exclude**', followed by a brief explanation of your choice.
                    """,
                },
            ]
        )
        
        # Return the content of the response
        return response['message']['content']
    
    except requests.exceptions.ConnectionError:
        return "Connection refused"
    
    except Exception as e:
        return f"Error: {e}"

