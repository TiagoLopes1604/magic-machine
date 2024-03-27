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


# Set page configuration
#st.set_page_config(page_title='Magic Machine', page_icon='🧙‍♂️')

st.set_page_config(
    page_title="Magic Machine",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
    background_color="#496C9F",  # Set your desired background color here
)

# Add title to the page
st.title('🧙‍♂️ Magic Machine')

# Add a sidebar with navigation links
page = st.sidebar.radio("Navigate", ["Calculate your treasure", "SkillSet"])

# Render different pages based on user selection
if page == "Calculate your treasure":
   # Add your content here
   st.markdown('**What can this app do?**')
   st.info('"Introducing Magic Machine, your go-to companion for navigating the dynamic world of data analytics careers! With Magic Machine, new data analysts can unlock the secrets to landing their dream job in this ever-evolving industry.Providing expert insights on industry trends, Magic Machine empowers aspiring data professionals to conquer the job market with confidence. Get ready to embark on your career journey with Magic Machine – where data meets destiny!')

   # Add audio file
   audio_file = open("PiratesOfTheCaribbeanThemeSong.mp3", "rb")
   audio_bytes = audio_file.read()
   st.audio(audio_bytes, format="audio/ogg")

   # Add GIF image
   with open("giphy.gif", "rb") as file:
       contents = file.read()
       data_url = base64.b64encode(contents).decode("utf-8")
       st.markdown(
           f'<img src="data:image/gif;base64,{data_url}" alt="gif" style="width: 700px; height: 600px;">', 
           unsafe_allow_html=True
       )
   
   # Add multiselect widget for selecting skills
   checkbox_labels = ['sql', 'python', 'excel', 'power_bi', 'tableau', 'sas', 'azure', 'snowflake', 'aws', 'spark', 'looker', 'qlik']
   st.write("<div style='text-align: center; font-size: 48px; font-weight: bold;'>Choose your skills</div>", unsafe_allow_html=True)
   selected_options = st.multiselect("", checkbox_labels)

   # Define function to predict salary based on selected skills
   def predict(selected_options):
       input_data = [[1 if label in selected_options else 0 for label in checkbox_labels]]
       input_df = pd.DataFrame(input_data, columns=checkbox_labels)
       input_df = input_df.astype(int)
       return input_df

   # Add button to trigger prediction
   if st.button("What's my bounty?"):
       if selected_options:
           input_df = predict(selected_options)
           prediction = model.predict(input_df)
           print_pred = str(np.round(prediction, 2))
           print_pred = print_pred.strip('[]')
           st.write(
               "<div style='text-align:center;'>"
               "<p style='font-size:70px;color:Black;'>Your Predicted Salary:</p>"
               f"<p style='font-size:125px;color:Red;display:inline;'>${print_pred}</p>"
               "</div>",
               unsafe_allow_html=True
           )
           with open("treasure.gif", "rb") as file:
               contents = file.read()
               data_url = base64.b64encode(contents).decode("utf-8")
               st.markdown(
                   f'<img src="data:image/gif;base64,{data_url}" alt="gif" style="width: 700px; height: 600px;">',
                   unsafe_allow_html=True
               )
       else:
           st.write("Please select at least one skill.")

elif page == "SkillSet":
    st.header('Find out what percentage of data analyst job offers you can cover with your skillset!')
    
    # Load data
    df = pd.read_csv('skills_occurences_income.csv')
    
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
    ## Skills selection
    skills_list = df.skill.unique()
    #skills_selection = st.multiselect('Select skills', skills_list, ['python', 'sql'])
    st.write("<div style='text-align: center; font-size: 48px; font-weight: bold;'>Choose your skills</div>", unsafe_allow_html=True)
    skills_selection =  st.multiselect('', skills_list)
    # Calculate and display total percentage for selected skills
    if skills_selection:
        total_percentage = calculate_total_percentage(skills_selection)
        st.write(f"The total percentage of selected skills is: {total_percentage:.2f}%")
        remaining_percentage = 100 - total_percentage
        st.write(f"The total percentage of remaining skills is: {remaining_percentage:.2f}%")
    
        bar_colors = ['#EB396A', '#65BCDA']
        # Calculate total
        total = total_percentage + remaining_percentage
        
        # Create a Plotly figure
        fig = go.Figure()
        
        # Add traces for each category
        fig.add_trace(go.Bar(x=['Total'], y=[total_percentage], name='Selected Skills', marker=dict(color=bar_colors[0])))
        fig.add_trace(go.Bar(x=['Total'], y=[remaining_percentage], name='Skills Still to Learn', marker=dict(color=bar_colors[1])))
        
        # Update layout
        fig.update_layout(
            title='Skills Overview',
            xaxis_title='Category',
            yaxis_title='Percentage',
            barmode='stack'
        )
        
        # Display the chart using Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        # Check if total_percentage is less than 30% to display the GIF
        if total_percentage < 30:
            # Open the GIF file in binary mode
            with open("study.gif", "rb") as file:
                # Read the contents of the file
                contents = file.read()
           
            # Encode the contents to base64
            data_url = base64.b64encode(contents).decode("utf-8")
           
            st.markdown(
               f'<img src="data:image/gif;base64,{data_url}" alt="gif" style="width: 700px; height: 600px;">', 
               unsafe_allow_html=True
            )
        else:
            # Open the other GIF file in binary mode
            with open("resp.gif", "rb") as file:
                # Read the contents of the file
                contents = file.read()
           
            # Encode the contents to base64
            data_url = base64.b64encode(contents).decode("utf-8")
           
            st.markdown(
               f'<img src="data:image/gif;base64,{data_url}" alt="gif" style="width: 700px; height: 600px;">', 
               unsafe_allow_html=True
            )
    else:
        st.write("Please select at least one skill.")






