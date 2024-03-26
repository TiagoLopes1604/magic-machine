import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile
import base64

# Page title
st.set_page_config(page_title='Magic Machine', page_icon='🧙‍♂️')
st.title('🧙‍♂️ Magic Machine')

st.markdown('**What can this app do?**')
st.info('"Introducing Magic Machine, your go-to companion for navigating the dynamic world of data analytics careers! With Magic Machine, new data analysts can unlock the secrets to landing their dream job in this ever-evolving industry.Providing expert insights on industry trends, Magic Machine empowers aspiring data professionals to conquer the job market with confidence. Get ready to embark on your career journey with Magic Machine – where data meets destiny!')

audio_file = open("PiratesOfTheCaribbeanThemeSong.mp3", "rb")
audio_bytes = audio_file.read()
st.audio(audio_bytes, format="audio/ogg")



import base64

# Open the GIF file in binary mode
with open("giphy.gif", "rb") as file:
    # Read the contents of the file
    contents = file.read()

# Encode the contents to base64
data_url = base64.b64encode(contents).decode("utf-8")

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="gif" style="width: 700px; height: 600px;">', 
    unsafe_allow_html=True
)


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile
import pickle



with open('model.pkl', 'rb') as f:
    model = pickle.load(f)



#df_input_skills = pd.read_csv('inputskills.csv')

#df_encode = pd.read_csv('df_encode.csv')
#df_opening_count = df_encode.groupby(['sql', 'python', 'excel', 'power_bi', 'tableau', 'sas', 'azure', 'snowflake', 'aws', 'spark', 'looker', 'qlik']).count()\
#[["ID"]].reset_index().rename(columns={"ID":"count"}).sort_values("count",ascending=False)
#df_input_skills = df_opening_count.iloc[ :,:12]
#df_output_percent = df_opening_count.iloc[ :,12:]

checkbox_labels = ['sql', 'python', 'excel', 'power_bi', 'tableau', 'sas', 'azure', 'snowflake', 'aws', 'spark', 'looker', 'qlik']
checkbox_states = {}
# Create a multiselect widget to select skills
selected_options = st.multiselect('Select skills:', checkbox_labels)

# Now you can pass the selected options to the predict function
def predict(selected_options):
    # Convert the selected skills to the input format required by the model
    input_data = [[1 if label in selected_options else 0 for label in checkbox_labels]]

    # Convert input_data to a DataFrame with the same structure as df_input_skills
    input_df = pd.DataFrame(input_data, columns=checkbox_labels)

    # Convert boolean values to integers
    input_df = input_df.astype(int)

    return input_df

    # Find the matching row in df_input_skills
    #matching_row = df_input_skills[df_input_skills.eq(input_df).all(axis=1)]

    # If a matching row is found, display the output
    #if not matching_row.empty:
       # matching_index = matching_row.index[0]
       # final_output = df_output_percent.loc[matching_index]
       # st.write("Job Opening Count:", final_output['count'])
       # st.write("Percentage of available openings:", f"{round(final_output['percentage'],2)} %")
   # else:
      #  st.write("No matching row found in the input data.")
   # return input_df


# Find the matching row in df_input_skills
#def postings():
    #input_data = [[1 if checkbox_states[label] else 0 for label in checkbox_labels]]

    # Convert input_data to a DataFrame with the same structure as df_input_skills
    #input_df = pd.DataFrame(input_data, columns=checkbox_labels)

    # Convert boolean values to integers
    #input_df = input_df.astype(int)
    #matching_row = df_input_skills[df_input_skills.eq(input_df).all(axis=1)]

    # If a matching row is found, display the output
    #if not matching_row.empty:
       #matching_index = matching_row.index[0]
       #final_output = df_output_percent.loc[matching_index]
       #st.write("Job Opening Count:", final_output['count'])
       #st.write("Percentage of available openings:", f"{round(final_output['percentage'],2)} %")
       #return final_output
    #else:
       #st.write("No matching row found in the input data.")
      # return None


# Add a button to trigger the prediction
if st.button('What am I worth?'):
    if selected_options:  # Check if any option is selected
        # Make predictions based on the selected skills
        input_df = predict(selected_options)

        # Perform prediction using the loaded model
        prediction = model.predict(input_df)
   
    
    ##openings = postings()
    
    # Display the prediction result
    st.write('Your Predicted Salary:', f"${print_pred}")

    # Calculate the image size based on the prediction value
    if isinstance(prediction, (int, float)):
        image_size = int(prediction) /1000  # Adjust the multiplier as needed
    else:
        image_size = 0  # Default image size if prediction is not a number

    # Read the GIF file
    try:
        with open("treasure.gif", "rb") as file:
            # Read the contents of the file
            contents = file.read()

        # Encode the contents to base64
        data_url = base64.b64encode(contents).decode("utf-8")

        # Embed the image in the app with the calculated size
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="gif" style="width: {image_size}px; height: {image_size}px;">', 
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error loading image: {e}")

with open("treasure.gif", "rb") as file:
            # Read the contents of the file
        contents = file.read()

        # Encode the contents to base64
        data_url = base64.b64encode(contents).decode("utf-8")

        # Embed the image in the app with the calculated size
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="gif" style="width: 500px; height: 500px;">', 
            unsafe_allow_html=True
        ) 
