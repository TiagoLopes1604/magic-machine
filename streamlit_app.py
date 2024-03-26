import streamlit as st
import pandas as pd
import numpy as np
import base64
import pickle

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the checkbox labels
checkbox_labels = ['SQL', 'Python', 'Excel', 'Power BI', 'Tableau', 'SAS', 'Azure', 'Snowflake', 'AWS', 'Spark', 'Looker', 'Qlik']

# Create a multiselect widget to select skills
selected_options = st.multiselect('Select skills:', checkbox_labels)

# Define a function to prepare input data for prediction
def predict(selected_options):
    input_data = [[1 if label in selected_options else 0 for label in checkbox_labels]]
    input_df = pd.DataFrame(input_data, columns=checkbox_labels)
    input_df = input_df.astype(int)
    return input_df

# Define a function to make predictions based on the selected skills
def make_prediction(selected_options):
    input_df = predict(selected_options)  # Prepare input data
    prediction = model.predict(input_df)  # Make prediction
    return prediction

# Add a button to trigger the prediction
if st.button('What am I worth?'):
    prediction = make_prediction(selected_options)  # Get the prediction

    # Display the predicted salary
    st.write('Your Predicted Salary:', f"${prediction}")

    # Calculate the image size based on the prediction value
    image_size = int(prediction) * 2  # Adjust the multiplier as needed

    # Load and display the GIF image
    with open("treasure.gif", "rb") as file:
        contents = file.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="gif" style="width: {image_size}px; height: {image_size}px;">', 
            unsafe_allow_html=True
        )
