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
import pickle
#import matplotlib.pyplot as plt
import plotly.graph_objs as go
# Page title
st.set_page_config(page_title='Magic Machine', page_icon='🧙‍♂️')
st.title('🧙‍♂️ Magic Machine')

#  Page title
st.title('Streamlit Multi-Page App')

# Add a sidebar with navigation links
page = st.sidebar.radio("Navigate", ["Home", "About"])

# Render different pages based on user selection
if page == "Home":
    st.header("Home Page")
    st.write("Welcome to the home page!")

elif page == "About":
    st.header("About Page")
    st.write("This is the about page. Here you can learn more about our app.")

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

#st.write("<div style='text-align: center; font-size: 48px; font-weight: bold;'>Job Market Insights</div>", unsafe_allow_html=True)
#st.image('jobs_per_week.PNG', caption='Jobs postings per week', width=700, use_column_width=False)
#st.image('slide3,2.PNG', caption='Jobs postings per weekday and per platform', width=700, use_column_width=False)
#st.image('slide4.PNG', caption='Jobs postings per location', width=700, use_column_width=False)

#st.write("<div style='text-align: center; font-size: 48px; font-weight: bold;'>Required Skills</div>", unsafe_allow_html=True)
#st.image('slide5a.PNG', caption='Number of mentions per skill', width=700, use_column_width=False)
#st.image('slide5b.PNG', caption='% by Category and Skill mentions over time', width=700, use_column_width=False)





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
        print_pred = str(np.round(prediction, 2))
        print_pred = print_pred.strip('[]')
   
    
    ##openings = postings()
    
    # Display the prediction result
    st.write('Your Predicted Salary:', f"${print_pred}")

   

with open("treasure.gif", "rb") as file:
            # Read the contents of the file
        contents = file.read()

        # Encode the contents to base64
        data_url = base64.b64encode(contents).decode("utf-8")

        # Embed the image in the app with the calculated size
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="gif" style="width: 700px; height: 600px;">', 
            unsafe_allow_html=True
        ) 

st.subheader('Find out what percentage of data anaylst job offers you can cover with your skillset!')
# Load data
df = pd.read_csv('skills_occurences_income.csv')
# df.year = df.year.astype('int')
# Function to calculate percentage
def calculate_percentage(skill_input):
    total_occurrences = df['count_occurrences'].sum()
    skill_occurrences = df.loc[df['skill'] == skill_input, 'count_occurrences'].iloc[0]
    skill_percentage = (skill_occurrences / total_occurrences) * 100
    return skill_percentage
# Function to calculate total percentage for multiple skills
def calculate_total_percentage(skills_input):
    total_percentage = 0
    for skill in skills_input:
        total_percentage += calculate_percentage(skill)
    return total_percentage
# Input widgets
## Genres selection
skills_list = df.skill.unique()
skills_selection = st.multiselect('Select skills', skills_list, ['python','sql'])
# Calculate and display total percentage for selected skills
if skills_selection:
    total_percentage = calculate_total_percentage(skills_selection)
    st.write(f"The total percentage of selected skills is: {total_percentage:.2f}%")
# Calculate and display remainig percentage for selected skills
if skills_selection:
    remaining_percentage = 100 - total_percentage
    st.write(f"The total percentage of remaining skills is: {remaining_percentage:.2f}%")
#plt.figure(figsize=(8, 6))
#plt.bar('Total', total_percentage, color='#EB396A', label='Selected Skills')
#plt.bar('Total', remaining_percentage, color='#65BCDA', label='Skills Still to Learn', bottom=total_percentage)
#plt.xlabel('Category')
#plt.ylabel('Percentage')
#plt.legend()
#plt.show()

# Define custom colors
bar_colors = ['#EB396A', '#65BCDA']

# Create a bar chart using Plotly
fig = go.Figure()
fig.add_trace(go.Bar(x=categories, y=total_percentage, name='Selected Skills', marker=dict(color=bar_colors[0])))
fig.add_trace(go.Bar(x=categories, y=remaining_percentage, name='Skills Still to Learn', marker=dict(color=bar_colors[1])))

# Update layout
fig.update_layout(
    title='Skills Overview',
    xaxis_title='Category',
    yaxis_title='Percentage',
    barmode='stack'
)

# Display the chart using Streamlit
st.plotly_chart(fig, use_container_width=True)



