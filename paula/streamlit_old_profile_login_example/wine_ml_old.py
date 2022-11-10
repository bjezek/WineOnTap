# Import required libraries
import streamlit as st
from datetime import datetime
from dataclasses import dataclass
from typing import Any, List
import pandas as pd
import hashlib

# check out week 18 from the bootcamp for example streamlit applications
# to run the file just cd into the directory where your files are stored
# and type 'streamlit run wine_ml.py'

# best documentation reference for streamlit_authenticator (login prompts)
# https://github.com/mkhorasani/Streamlit-Authenticator

# documentation on forms (for user preferences)
# https://blog.streamlit.io/introducing-submit-button-and-forms/

# Needed for log in prompts
import streamlit_authenticator as stauth
# https://towardsdatascience.com/how-to-add-a-user-authentication-service-in-streamlit-a8b93bf02031
# pip install streamlit-authenticator

# Needed to store user credentials (I followed how this was done in the link below the import statement for streamlit_authenticator)
import yaml
# https://python.land/data-processing/python-yaml
# pip install pyyaml

# the config (.yaml) file stores user credentials
with open('config.yaml') as file:
    config = yaml.safe_load(file) # from yaml link above
    
# welcome statement

#st.markdown('<p class="font">Guess the object Names</p>', unsafe_allow_html=True)
#https://python.plainenglish.io/three-tips-to-improve-your-streamlit-app-a4c94b4d2b30

st.markdown(""" <style> .font {
font-size:50px ; font-family: 'Times New Roman'; color: #000000;} 
</style> """, unsafe_allow_html=True)

st.markdown('<p class="font">Welcome to WINE-ML!</p>', unsafe_allow_html=True)
#st.write("Welcome to WINE-ML! Please log in to continue to your personalized recommendations:")

# https://unsplash.com/s/photos/wine

from PIL import Image
image = Image.open('unsplashimage2.jpg')

st.image(image, caption='Photo by Kym Ellis on Unsplash')

#Photo by <a href="https://unsplash.com/@kymellis?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Kym Ellis</a> on <a href="https://unsplash.com/s/photos/wine?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  

# pull credentials from config yaml file
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# display log-in panel for user log in
name, authentication_status, username = authenticator.login('Login', 'main')

# display prompts (if authentication_status = True, they are welcomed, if it's False, states user/pasw incorrect, if it's empty, asks user to log in )
if authentication_status:
    authenticator.logout('Logout', 'main')
    st.title(f'Welcome *{name}*')
    st.write('Your preferences profile:')
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')

# setting default preferences 
# need to update these
state_input = "Texas"
city_input = "Houston"
age_input = "50"
redwhite_input = "red"
cheap_or_expensive_input = "cheap"
range_low_input = "0"
range_high_input = "50"
rose_input = "no"
merlot_input = "no"
cab_input = "yes"
chardonnay_input = "no"
misc_like_input = "no"
misc_dislike_input = "no"
misc_text = "leave me alone"
aroma_input = ""
flavor_input = ""
pairing_input = ""

# shows the user what the default preferences are
# need to update these
st.write("Please update your preferences by using the button above if these tastes aren't to your liking:")
st.write("Red or white?")
st.write(redwhite_input)
st.write("Merlot?")
st.write(merlot_input)

# shows the user what their profile is based on
# need to update these
st.write("Here are your personal details that are also used to inform our recommendation:")
state_input = "Texas"
city_input = "Houston"
age_input = "50"

st.checkbox("hello", value=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False)

# if the user is authenticated (logged in), give them a button to update profile details
if authentication_status:
    st.write("Please update your preferences profile so we can deliver you personalized recommendations!")
    if st.button("Update preferences profile"):
        with st.form(key='my_form'):
            
            state_input = st.text_input(label='What state do you currently reside in?')
            city_input = st.text_input(label='What city do you currently reside in?') 
            age_input = st.text_input(label='How old are you?')
            redwhite_input = st.text_input(label='Red or white?')
            cheap_or_expensive_input = st.text_input(label = "Cheap or expensive?")
            range_low_input = st.text_input(label = "What's the lowest price per bottle that you'd pay?")
            range_high_input = st.text_input(label = "What's the highest price per bottle that you'd pay?")
            rose_input = st.text_input(label='Do you like rose?')
            merlot_input = st.text_input(label='Do you like merlot?')
            cab_input = st.text_input(label='Do you like cabs?')
            chardonnay_input = st.text_input(label='Do you like chardonnay?')
            aroma_input = st.text_input(label='Any particular aromas that you\'re looking for?')
            flavor_input = st.text_input(label='Any particular flavors that you\'re looking for?')
            pairing_input = st.text_input(label='What pairing are you looking for?')
            misc_like_input = st.text_input(label='Please input any other wines that you like!')
            misc_dislike_input = st.text_input(label='Please input any other wines that you dislike or try to avoid!')
            misc_text = st.text_input(label='Is there anything else that you want our models to know about you?')
            
            submit_button = st.form_submit_button(label='Submit')

df = pd.read_csv("amazon_data_largest_reviews_and_ratings.csv")

#works but doesn't display links
#st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

import streamlit as st 

linked = df["Link"][1]

comment = f"check out this [link]({linked})"

# was this a good recommendation for you? 
# dump to config file
# if it wasn't a good recommendation, maybe try something else next time? 

st.write(comment)
#works
#st.write("check out this [link](https://www.amazon.com/19-Crimes-Red-Blend-750/dp/B01N7CZQJX/ref=sr_1_22_f3_0o_wf?keywords=Cabernet+Sauvignon&qid=1667766734&refinements=p_72%3A1248897011&rnid=1248895011&s=wine&sr=1-22)")

# prompt user to fill out form to register for an account if they haven't done so already
st.write("")
st.write("Fill out the form below to register for an account!")

# if the user fills out the form and hits submit, the information is written back to the config yaml file
try:
    if authenticator.register_user('Register user', preauthorization=False):
        st.success('User registered successfully')
        with open('config.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
except Exception as e:
    st.error(e)