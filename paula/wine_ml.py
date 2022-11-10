# import statements

import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image

import pickle

# import statements for random forest model (for wine quality prediction)
# import dataset
wine_red = pd.read_csv('winequality-red.csv')
# create scaler function for use later in the application
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
normal_wine = scaler.fit_transform(wine_red)
normal_wine = pd.DataFrame(normal_wine, columns = wine_red.columns)
# import random forest model
loaded_quality_model_rf = pickle.load(open('quality_model_rf_11_10.sav', 'rb'))

#input statements for neural network wine review data  (for wine quality prediction)
# import dataset
df_reviews = pd.read_csv("winemag-data-130k-v2.csv")
# import NN model
# TO DO: add .h5 model here
# TO DO: add import statements for NN model

# Welcome page
def intro():

    # Greet the user
    st.write("# Welcome to Wine-ML!")

    # Display image 
    image = Image.open('unsplashimage2.jpg')
    st.image(image, caption='Photo by Kym Ellis on Unsplash')
    #Photo by <a href="https://unsplash.com/@kymellis?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Kym Ellis</a> on <a href="https://unsplash.com/s/photos/wine?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
    
    # Create sidebar selector
    st.sidebar.success("Select a demo above.")

    # Tell the user what the application can do
    st.markdown(
        """
        ## Trying to find a good wine to try tonight and can't make a decision?
        ## Want to find another wine similar to your favorites? 
        ## Then Wine-ML is for you!

        **👈 Make a selection from the dropdown on the left** to see how our
        models can work for you! 
        
        """
    )

# Wine quality calculator demonstration
def quality_demo():

    # Header
    st.write("# Wine Quality Calculator")
    
    # Instruct the user on how to use the application
    st.write(
        """
        Use our models to help you pick great quality wines. Our models indicate
        that alcohol content, volatile acidity, and sulphates are the most important variables when it comes to 
        dictating the quality of a wine.
        
        Don't feel the need to stick to wines that have been reviewed when you can use
        our models to help you gauge its quality without having to taste it. 

        Enter as much information that you have about the wine below, and use our
        models to help you predict its quality!
        
    """
    )

    #Streamlit documentation reference
    #https://discuss.streamlit.io/t/how-to-take-text-input-from-a-user/187
    
    # Create input section for the user to input parameters
    fixed_acidity = st.number_input("Fixed acidity", min_value = 4.6, max_value = 15.9, value = 8.32, step = 0.01)
    volatile_acidity = st.number_input("Volatile acidity", min_value = 0.12, max_value = 1.58, value = 0.53, step = 0.01)
    citric_acid = st.number_input("Citric acid", min_value = 0.0, max_value = 1.0, value = 0.27, step = 0.01)
    residual_sugar = st.number_input("Residual sugar", min_value = 0.9, max_value = 15.5, value = 2.54, step = 0.01)
    chlorides = st.number_input("Chlorides", min_value = 0.01, max_value = 0.61, value = 0.09, step = 0.01)
    free_sulfur_dioxide = st.number_input("Free sulfur dioxide", min_value = 1.0, max_value = 72.0, value = 15.87, step = 0.01)
    total_sulfur_dioxide = st.number_input("Total sulfur dioxide", min_value = 6.0, max_value = 289.0, value = 46.47, step = 0.01)
    density = st.number_input("Density", min_value = 0.99, max_value = 1.0, value = 1.0, step = 0.01)
    ph = st.number_input("pH", min_value = 2.74, max_value = 4.01, value = 3.31, step = 0.01)
    sulphates = st.number_input("Sulphates", min_value = 0.33, max_value = 2.0, value = 0.66, step = 0.01)
    alcohol = st.number_input("Alcohol", min_value = 8.4, max_value = 14.9, value = 10.42, step = 0.01)
    
    # wrap the input parameters into a dataframe for use with the models

    quality_input = {
        'fixed acidity': [fixed_acidity], 
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [ph],
        'sulphates': [sulphates],
        'alcohol': [alcohol],
        }
    
    input_for_quality_pred = pd.DataFrame.from_dict(quality_input)
    input_for_quality_pred_full = pd.concat([input_for_quality_pred, wine_red.drop(["quality"], axis = 1)])

    # scale the data (keep in mind first row is user input data)
    input_for_quality_pred_scaled_full = scaler.fit_transform(input_for_quality_pred_full)
    input_for_quality_pred_scaled_full = pd.DataFrame(input_for_quality_pred_scaled_full, columns = input_for_quality_pred.columns)
   
    # make the prediction for the user
    pred_quality = loaded_quality_model_rf.predict(input_for_quality_pred_scaled_full)[0]
    st.write("The estimated quality of a wine with the parameters you selected is:")

    # convert model output into user-readable text
    if pred_quality == "no":
        pred_quality_comment = "The parameters input correspond to a wine with LOW quality (rating of 6 or lower)"
    else:
        pred_quality_comment = "The parameters input correspond to a wine with HIGH quality (rating of 7 or greater)"
    
    # output the results from the model
    st.write(pred_quality_comment)


# create demonstration for neural network model
def reviews_demo():
    # header
    st.write("# Wine Quality Predictor")
    
    # instruct the user on how to use the application
    st.write(
        """
        Use our models to help you predict the quality of wine based on its reviews!

        Enter as much information that you have about the wine below, and use our
        models to help you predict its quality!

        The form is already pre-filled with an example,
        make sure to DELETE any information that's not relevant for your wine!
        
    """
    )
    
    # choose the row to display for the example data for the user
    row = 4

    # create input fields for the user
    country = st.text_input("Country", value=df_reviews.iloc[4,:]["country"])
    description = st.text_input("Description", value=df_reviews.iloc[row,:]["description"])
    designation = st.text_input("Designation", value=df_reviews.iloc[row,:]["designation"])
    price = st.number_input("Price", min_value = 0.0, max_value = 1000000.00, value = df_reviews.iloc[row,:]["price"], step = 0.01)
    province = st.text_input("Province", value=df_reviews.iloc[row,:]["province"])
    region_1 = st.text_input("Region 1", value=df_reviews.iloc[row,:]["region_1"])
    region_2 = st.text_input("Region 2", value=df_reviews.iloc[row,:]["region_2"])
    #region_2 = st.text_input("Country", value="") #what's the difference between region 1 and 2?
    taster_name = st.text_input("Taster Name", value=df_reviews.iloc[row,:]["taster_name"])
    taster_twitter_handle = st.text_input("Taster Twitter Handle", value=df_reviews.iloc[row,:]["taster_twitter_handle"])
    title = st.text_input("Title", value=df_reviews.iloc[row,:]["title"])
    variety = st.text_input("Variety", value=df_reviews.iloc[row,:]["variety"])
    winery = st.text_input("Winery", value=df_reviews.iloc[row,:]["winery"])

    # wrap the user input data into a dataframe for the model
    
    review_input = {
        'country': [country], 
        'description': [description],
        'designation': [designation],
        'price': [price],
        'province': [province],
        'region_1': [region_1],
        'region_2': [region_1],
        'taster_name': [taster_name],
        'taster_twitter_handle': [taster_twitter_handle],
        'title': [title],
        'variety': [variety],
        'winery': [winery],
        }

    input_for_review_pred = pd.DataFrame.from_dict(review_input)
    # if needed, combine the user input data with the historical dataset (? might need this for scaling purposes?)
    input_for_review_pred_full = pd.concat([input_for_review_pred, df_reviews.drop(["points"], axis = 1)])

    ##TO DO need to figure out how to handle scaler with text input
    #input_for_review_pred_scaled_full = scaler.fit_transform(input_for_quality_pred_full)
    #input_for_review_pred_scaled_full = pd.DataFrame(input_for_review_pred_scaled_full, columns = input_for_review_pred.columns)

    ##TO DO need to handle empty values
    ##should we just set any empty values to the mean of the dataset? 
    
    #TO DO need to update this with loaded NN .h5 file from Jerome
    #review_quality = loaded_review_model_nn.predict(input_for_review_pred_scaled_full)[0]

    #TO DO adjust comments for the user based on outputs from the NN model
    # #adjust based on output of nn model
    # if pred_quality == "no":
    #     pred_quality_comment = "The parameters input correspond to a wine with LOW quality (rating of 6 or lower)"
    # else:
    #     pred_quality_comment = "The parameters input correspond to a wine with HIGH quality (rating of 7 or greater)"
    
    
    #show the user what the results from the model are
    st.write("Based on your review, the estimated quality of this wine is: ")


# Dictionary for sidebar
page_names_to_funcs = {
    "Welcome!": intro,
    "Wine Quality Calculator": quality_demo,
    "Wine Quality Predictor": reviews_demo,
    #"Other": data_frame_demo
}

# sidebar selector
demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()