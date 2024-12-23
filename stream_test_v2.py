
import streamlit as st
import pandas as pd
import plotly.express as px
import re
import numpy as np
import pandas as pd
import pandas as pd
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load, generate
from mlx_lm.tuner import train, evaluate, TrainingArgs
from mlx_lm.tuner import linear_to_lora_layers
from pathlib import Path
from tqdm import tqdm
from Prompts.llama3_prompts_Screening_lama1B_v1 import Screening_v1
from Prompts.llama3_prompts_Screening_mistral_v4 import Screening_mistral_v4
import ollama
import json


# Set the page configuration
st.set_page_config(page_title="Streamlit App with Logos", layout="wide")
# Create a layout for logos
logo_col1, logo_col2, logo_col3, logo_col4 = st.columns(4)
logo_width = 100  # Adjust this value as needed
with logo_col1:
    st.image("images/udem.png", width= logo_width)
with logo_col2:
    st.image("images/espum.jpeg", width= logo_width)
with logo_col3:
    st.image("images/solidarité.png", width= logo_width)
with logo_col4:
    st.image("images/cresp.png", width= logo_width)

# Add logos at the top of the page
st.markdown("### Contribution of artificial intelligece for people with disabilities")
st.markdown("---")
dim_set = ['objectives', 'AI_application', 'Disability_type', 'Barriers', 'Facilitators', 'Equity',  
           'Tools', 'Disability_outcome', 'Implications']



# Sidebar Navigation
st.sidebar.title("Menu")
def clean_text(text):
    """Clean the text by removing extra spaces and newlines."""
    return re.sub(r'\s+', ' ', text.replace('\n', ' '))

menu = st.sidebar.radio("Select a Step", ["Phase 1: Title and abstract screening",
 "Phase 2: Full text screening", "Phase 3: Data Extraction"])

# Step 1: Upload File
if menu == "Phase 1: Title and abstract screening":
    st.markdown("### Phase 1: Title and Abstract Screening")
    uploaded_file = st.file_uploader("Upload Your Excel File Containing Titles and Abstracts", type=["xlsx", "xls"])

    if uploaded_file is not None:
        # Load the Excel file into a DataFrame
        df = pd.read_excel(uploaded_file)
        st.markdown(f"<p style='color:green;'>{len(df)} records uploaded successfully!</p>", unsafe_allow_html=True)

        # Allow the user to select the columns for Title and Abstract
        title_column = st.selectbox("Select the column for Titles", df.columns)
        abstract_column = st.selectbox("Select the column for Abstracts", df.columns)

        if title_column and abstract_column:
            # Combine selected columns
            df['Combined'] = df[title_column] + " " + df[abstract_column]
            df['Combined'] = df['Combined'].astype(str)
            df['Combined'] = df['Combined'].apply(clean_text)
            tqdm.pandas()

            # Add a button for starting the processing
            if st.button("Start Reviewing Titles and Abstracts"):
                with st.spinner("Processing titles and abstracts..."):
                    df['result'] = df['Combined'].progress_apply(Screening_v1)

                # Display processed data
                st.success("Processing completed!")
                st.write("Processed Data Preview:", df[[title_column, abstract_column, 'result']].head())

                # Optionally download the processed data
                @st.cache_data
                def convert_df_to_csv(dataframe):
                    return dataframe.to_csv(index=False).encode('utf-8')

                csv = convert_df_to_csv(df)
                st.download_button(
                    label="Download Processed Data",
                    data=csv,
                    file_name='processed_data.csv',
                    mime='text/csv',
                )
        else:
            st.error("Please select valid columns for Title and Abstract.")

# Step 2: Visualization
elif menu == "Phase 2: Full text screening":
    st.markdown("### Phase 2: Full-Text Screening")
    uploaded_file = st.file_uploader("Upload Your Excel File", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            # Load the Excel file into a DataFrame
            df = pd.read_excel(uploaded_file)
            # st.write("Uploaded Data Preview:")
            # st.dataframe(df)
            st.markdown(f"<p style='color:green;'>{len(df)} Full-texts Entered</p>", unsafe_allow_html=True)

            # Save the DataFrame in Streamlit's session state for later use
            st.session_state["df"] = df

            # Allow the user to select the columns for Text
            text_col = st.selectbox("Select the column for Titles", df.columns)
            # Check if the required 'text' column exists
            if text_col in df.columns:
                # Button to start processing
                if st.button("Start Full-Text Screening"):
                    with st.spinner("Processing full-text screening..."):
                        df['result'] = df['text'].progress_apply(Screening_mistral_v4)

                    # Display results
                    st.success("Full-text screening completed!")
                    st.write("Processed Data Preview:")
                    st.dataframe(df[['text', 'result']].head())

                    # Optionally download the processed data
                    @st.cache_data
                    def convert_df_to_csv(dataframe):
                        return dataframe.to_csv(index=False).encode('utf-8')

                    csv = convert_df_to_csv(df)
                    st.download_button(
                        label="Download Processed Data",
                        data=csv,
                        file_name='processed_full_text_screening.csv',
                        mime='text/csv',
                    )
            else:
                st.error("The uploaded file must contain a 'text' column.")
        except Exception as e:
            st.error(f"Error loading the file: {e}")
    else:
        st.info("Please upload an Excel file to proceed.")
 
# Step 3: Data Extraction
elif menu == "Phase 3: Data Extraction":
    st.markdown("### Phase 3: Data Extraction")

# File Upload Section
    uploaded_file = st.file_uploader("Upload your result file here", type=["xlsx", "xls"])
    def load_model(dimension):
        model_path = "mistralai/Mistral-7B-Instruct-v0.2"
        adapter_path = Path(f"Models/adapters_{dimension}")  # Convert to Path object
    
    # Define LoRA configuration
        lora_config = {
        "lora_layers": 8,
        "num_layers": 12,  # Replace with the actual number of layers for your model
        "lora_parameters": {
            "rank": 8,
            "scale": 10.0,
            "dropout": 0.1,
            },
        }

    # Save the LoRA config to the adapter path
        adapter_path.mkdir(parents=True, exist_ok=True)
        with open(adapter_path / "adapter_config.json", "w") as fid:
            json.dump(lora_config, fid, indent=4)

    # Load model and tokenizer using the load function
        model, tokenizer = load(
            path_or_hf_repo=model_path,
            adapter_path=adapter_path
        )
        return model, tokenizer

    # Instructions for each dimension
    instructions = {
        "objectives": "[INST]You are an expert in objective extraction from summary of papers. What is the objective of the following full text? Just tell me in one sentence and do not repeat your answer and summarize your answer, do not give me unfinished sentences or sentences related to method, I only want objective of the paper:",
        "AI_app": "[INST]You are an expert in data extraction of systematic reviews, What is the AI application used for the disable population in the following full-text based on this article?(for example predictions, decision-making, physical-aide, classification, clustering), just name the application from options that i gave to you in one word:",
        "Disability_type": "[INST] You are an expert in data extraction of systematic reviews, given the following which disability impairement does this text works on (between vision, cognition, communication, auditive, and motor)? Just select from the options I gave to you: ",
        "Barriers": "[INST] You are an expert in data extraction of systematic reviews, given the following which disability impairement does this text works on (between vision, cognition, communication, auditive, and motor)? Just select from the options I gave to you: ",
        "Facilitators" : "[INST] You are an expert in data extraction of systematic reviews, what is the facilitators of of the following full-text to inlcude people with handicap in text based on this article?, for example I have considered these items as the facilitators in the studies if they: Extract information on the facilitators for including people with disabilities in AI research mentioned in each paper, such as: + Collaboration with disability organizations and advocates + Training and expertise in accessibility and inclusivity + Adaptive data collection methods (e.g., online surveys) + Inclusive participant recruitment strategies + Facilitate interacting with computers by providing an assistive devices + Using technology for detection of problems among people with disability just give it in one short sentence no additional words, if they have not addressed the facilitatiors just say : not addressed" ,
        "Equity" : "[INST] You are an expert in data extraction of systematic reviews, Is equity considered in the following full-text to inlcude people with handicap in  text based on this article?,for example I have considered these items as considering equity: 1. Diversity in sample consideration 2. Promoting equity in access to technology and computer interaction 3. Including nation-wide registry as the sample data 4. Involving usability testing with people with disabilities to ensure that AI solutions meet their needs. 5. Help in developing policies to improve the health of this vulnerable population. 6. Ensuring that all patients have equitable access to the technology. just give it in one short sentence no additional words",
        "Tools" : "[INST] You are an expert in data extraction of systematic reviews, Extract information on the specific applications or tools developed using AI that aim to improve the lives of persons with disabilities in the following full-text  This could include, but is not limited to: + Assistive technologies for daily living tasks (e.g., automated door openers, smart home devices) + Communication aids (e.g., speech-to-text systems, sign language recognition) + Diagnostic tools (e.g., image analysis software for visual impairments, audio processing software for hearing impairments) + Rehabilitation and therapy tools (e.g., virtual reality-based physical therapy, AI-powered cognitive training) just give it in one short phrase no additional explanations",
        "Disability_outcome" : "[INST] You are an expert in data extraction of systematic reviews,Extract information on disability-specific outcomes mentioned in the following full-textsuch as improved communication skills for individuals with speech impairments but not limited to this, search it based the text I gave you. just give it in one short phrase no additional explanations",
        "Implications" : "[INST] You are an expert in data extraction of systematic reviews, what is the Implications of this study to inlcude people with handicap in the following full-text based on this article?, just give it in one short sentence no additional words"
    }

    if uploaded_file is not None:
    # Load the Excel file into a DataFrame
        try:
            df = pd.read_excel(uploaded_file)

        # Show the uploaded data
            st.write("Uploaded Data:")
            st.markdown(f"<p style='color:green;'>{len(df)} Full-texts Entered</p>", unsafe_allow_html=True)
            # User selects dimensions to extract
            selected_dimension = st.selectbox(
                "Select Dimensions for Extraction",
                options=dim_set,
                help="Choose the dimension to extract data using the corresponding models."
            )
            instruction = instructions[selected_dimension]
            # Validate if a 'text' column exists
            if "text" in df.columns:
                if st.button("Start Dimension Extraction"):
                    with st.spinner("Extracting data for selected dimensions..."):
                        # Process for each selected dimension
                        model, tokenizer = load_model(selected_dimension)

                        def generate_text(text):
                                prompt = instruction + text + "[/INST]"
                                return generate(model, tokenizer, prompt, max_tokens=200)

                        df[f"{selected_dimension}_output"] = df["text"].apply(generate_text)


                    # Display results
                    st.success("Dimension extraction completed!")
                    st.write("Processed Data Preview:")

                    # Ensure selected_dimension is treated as a list during concatenation
                    st.dataframe(df[['text'] + [selected_dimension]].head())

                    # Optionally download the processed data
                    @st.cache_data
                    def convert_df_to_csv(dataframe):
                        return dataframe.to_csv(index=False).encode('utf-8')

                    csv = convert_df_to_csv(df)
                    st.download_button(
                         label="Download Processed Data",
                        data=csv,
                        file_name='processed_dimension_data.csv',
                        mime='text/csv',
                    )

        # Column Selection
            country_column = st.selectbox("Select the 'Country' Column", df.columns)

        # Plotting the World Map
            if country_column:
            # Count the number of occurrences for each country
                country_counts = df[country_column].value_counts().reset_index()
                country_counts.columns = ["Country", "Count"]

            # Create the world map
                fig = px.choropleth(
                    country_counts,
                    locations="Country",
                    locationmode="country names",
                    color="Count",
                    title="World Map of Studies",
                    color_continuous_scale=px.colors.sequential.Turbo,
                    width=1000,  # Adjust the width
                    height=600,  # Adjust the height
                )

            # Display the map
                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error loading the file: {e}")
else:
    st.info("Please upload an Excel file to proceed.")


