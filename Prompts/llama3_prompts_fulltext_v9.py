import pandas as pd
import requests
import ollama
from json import loads, dumps

# Disability categories screening function:
def fulltext_screening_v10(text):
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

Your task is to determine whether the paper should be **Include**d or **Exclude**d for a full-text review, based strictly on these criteria:

---

**Include** only if:
1. The paper explicitly focuses on Artificial Intelligence (AI) or machine learning (ML) applications in diagnostic tools, predictive tools or clinical trials **to benefit people with disabilities**.
2. The AI/ML models used in the study include:
   - **Supervised learning models** (e.g., classification, regression, survival analysis) for people with disabilities.
   - **Unsupervised learning models** (e.g., clustering, dimensionality reduction) for people with disabilities.
   - **Deep learning models** (e.g., CNNs, RNNs, transformers) for people with disabilities.
   - **Natural Language Processing (NLP) models** (e.g., named entity recognition, sentiment analysis, text summarization) for people with disabilities.
   - Predictive models for disability-related outcomes (e.g., identifying high-risk populations, progression risk) for people with disabilities.
   - Applications of AI/ML in clinical decision support for disabilities.

---

**Exclude** if:
1. The paper **does not explicitly focus on AI/ML applications** for disability-related interventions or outcomes or for people with disabilities.
2. The study is about:
   - Assistive devices without AI/ML (e.g., wearable devices, Bluetooth, rehabilitation tools).
   - Feasibility studies, pilot studies, or statistical analysis without reporting AI/ML model performance.
   - Reviews, meta-analyses, book chapters, conference papers, or posters.
3. The focus is on disease prediction algorithms unrelated to disability-specific outcomes or interventions.

If you cannot confidently decide, default to "**Exclude**".

---

Respond with **strictly one word**: "**Include**" or "**Exclude**". 
After the decision, provide a brief explanation of your choice in one sentence.
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
