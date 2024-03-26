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
