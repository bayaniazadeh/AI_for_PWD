
import streamlit as st
import pandas as pd
import plotly.express as px
import re
from tqdm import tqdm
from Prompts.llama3_prompts_Screening_lama1B_v1 import Screening_v1
from Prompts.llama3_prompts_Screening_mistral_v4 import Screening_mistral_v4
import ollama


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

    if uploaded_file is not None:
    # Load the Excel file into a DataFrame
        try:
            df = pd.read_excel(uploaded_file)

        # Show the uploaded data
            st.write("Uploaded Data:")
            st.markdown(f"<p style='color:green;'>{len(df)} Full-texts Entered</p>", unsafe_allow_html=True)
            # User selects dimensions to extract
            selected_dimensions = st.multiselect(
                "Select Dimensions for Extraction",
                options=dim_set,
                default=[],
                help="Choose one or more dimensions to extract data using the corresponding models."
            )
            # Validate if a 'text' column exists
            if "text" in df.columns:
                if st.button("Start Dimension Extraction"):
                    with st.spinner("Extracting data for selected dimensions..."):
                        # Process for each selected dimension
                        for dimension in selected_dimensions:
                            model = f"model_{dimension}"  # Dynamically map model names
                            st.info(f"Processing dimension: {dimension}")
                            # df[dimension] = df['text'].progress_apply(
                            #     lambda text: model(text, model)
                            # )

                    # Display results
                    st.success("Dimension extraction completed!")
                    st.write("Processed Data Preview:")
                    st.dataframe(df[['text'] + selected_dimensions].head())

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


