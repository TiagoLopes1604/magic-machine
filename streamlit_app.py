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



df_input_skills = pd.read_csv('inputskills.csv')

df_encode = pd.read_csv('df_encode.csv')
df_opening_count = df_encode.groupby(['sql', 'python', 'excel', 'power_bi', 'tableau', 'sas', 'azure', 'snowflake', 'aws', 'spark', 'looker', 'qlik']).count()\
[["ID"]].reset_index().rename(columns={"ID":"count"}).sort_values("count",ascending=False)
df_input_skills = df_opening_count.iloc[ :,:12]
df_output_percent = df_opening_count.iloc[ :,12:]

checkbox_labels = ['sql', 'python', 'excel', 'power_bi', 'tableau', 'sas', 'azure', 'snowflake', 'aws', 'spark', 'looker', 'qlik']

# Create a dictionary to store the checkbox states
checkbox_states = {}

# Create checkboxes for each label and store their states in the dictionary
for label in checkbox_labels:
    checkbox_states[label] = st.checkbox(label)



# Define a function to make predictions based on the selected checkboxes
def predict():
    # Convert the selected checkboxes to the input format required by the model
    input_data = [[1 if checkbox_states[label] else 0 for label in checkbox_labels]]

    # Convert input_data to a DataFrame with the same structure as df_input_skills
    input_df = pd.DataFrame(input_data, columns=checkbox_labels)

    # Convert boolean values to integers
    input_df = input_df.astype(int)

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
    return input_df


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
    input_df = predict()
    

    prediction = model.predict(input_df)
    print_pred = str(np.round(prediction, 2))
    print_pred = print_pred.strip('[]')
    
    ##openings = postings()
    
    # Display the prediction result

    st.write('Your Predicted Salary:', f"${print_pred}")
    #st.write('Your Predicted Salary:', f"${np.round(prediction, 2)}")
    #st.write("Percentage of available openings:", f"{round(openings['percentage'],2)} %")

