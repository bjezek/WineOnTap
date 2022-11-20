# import statements

# general
import streamlit as st
import pandas as pd
import numpy as np
import re

# for plotting
import seaborn as sns
import matplotlib.pyplot as plt

# for image display in the streamlit application
from PIL import Image

# for random forest model export/import
import pickle

#from jerome's & gautam's files for neural network modeling / nlp modeling
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, metrics
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
wine_data = pd.read_csv("wine_mag-data_first-150k.csv", index_col = False)


# import statements for brandon's random forest model (for wine quality prediction)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
# red wine dataset
wine_red = pd.read_csv('winequality-red.csv')
normal_wine = scaler.fit_transform(wine_red)
normal_wine = pd.DataFrame(normal_wine, columns = wine_red.columns)
loaded_quality_model_rf = pickle.load(open('quality_model_rf_11_10.sav', 'rb'))
# white wine dataset
wine_white = pd.read_csv('winequality-white.csv')
normal_wine_1 = scaler.fit_transform(wine_white)
normal_wine_1 = pd.DataFrame(normal_wine_1, columns = wine_white.columns)
loaded_quality_model_rf_1 = pickle.load(open('quality_model_rf_white.sav', 'rb'))

# import wine review data for use later in the application
df_reviews = pd.read_csv("winemag-data-130k-v2.csv")

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
    # Instruct the user on how to use the application
    st.write("# Wine Quality Calculator")
       
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
    color = st.radio("What color of wine is this for?",('red', 'white'), index = 0)
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
    
    # associated code and files are taken from brandon's random forest modeling with the dataset
    
    # determine which dataset to use (red or white):
    if color == "red":
        wine_color = wine_red.copy()
        prediction_model = loaded_quality_model_rf
    elif color == "white":
        wine_color = wine_white.copy()
        prediction_model = loaded_quality_model_rf_1
    else:
        print("Error on wine color with streamlit radio buttons")

    # combine the input data with the testing dataset
    input_for_quality_pred = pd.DataFrame.from_dict(quality_input)
    input_for_quality_pred_full = pd.concat([input_for_quality_pred, wine_color.drop(["quality"], axis = 1)])

    # scale the data (keep in mind first row is user input data)
    input_for_quality_pred_scaled_full = scaler.fit_transform(input_for_quality_pred_full)
    input_for_quality_pred_scaled_full = pd.DataFrame(input_for_quality_pred_scaled_full, columns = input_for_quality_pred.columns)
   
    # make the prediction for the user
    pred_quality = prediction_model.predict(input_for_quality_pred_scaled_full)[0]
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

        Enter a review about a wine, and use our
        models to help you predict its quality!

        The form is already pre-filled with an example,
        make sure to DELETE any information that's not relevant for your wine!
        
    """
    )
    
    # choose the row to display for the example data for the user
    row = 4

    # create input field for the user
    description = st.text_input("Description", value=df_reviews.iloc[row,:]["description"])
    
    # other variable assignments that are not used but could be used in the future include:
    country = ""
    designation = "" 
    price = "" 
    province = "" 
    region_1 = "" 
    region_2 = "" 
    taster_name = "" 
    taster_twitter_handle = "" 
    title = "" 
    variety = "" 
    winery = "" 

    # wrap the user input data into a dataframe for model input
    #
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

    # If the user clicks the "calculate" button, use the neural network model to calculate quality of the wine
    # The code to replicate the neural network is copied from the code that was provided in Jerome's notebook
    # The .h5 model that was incorporated into this notebook was exported from Jerome's workbook
    # The user input is captured as a dataframe and then appended with the testing dataset.   
    
    if st.button("Calculate!"): 
        #show the user what the results from the model are
        st.write("Calculating model...")
        

        #create a dataframe from the user input
        #(note that "description" is the only field that's populated and used for the model)
        input_for_review_pred = pd.DataFrame.from_dict(review_input)

        #nltk.download('stopwords')
        #https://python-forum.io/thread-31052.html
        #reference stopwords locally
        from nltk.corpus import stopwords #saved
        stopwords = set(stopwords.words('english'))
        detokenizer = TreebankWordDetokenizer()

        #Remove all stop words and 
        def cleaning_words(description):
            description = word_tokenize(description.lower())
            description = [token for token in description if token not in stopwords and token.isalpha()]
            return detokenizer.detokenize(description)

        wine_data["new_desc"] = wine_data["description"].apply(cleaning_words)
        #apply cleaning_words to user input as well
        input_for_review_pred["new_desc"] = input_for_review_pred["description"].apply(cleaning_words)
        
        review_train, review_test, y_train, y_test = train_test_split(wine_data["new_desc"], wine_data["points"], test_size=0.25, random_state=102300)
        
        # fit tokenizer on training dataset
        tokenizer = Tokenizer(num_words=7000)
        tokenizer.fit_on_texts(review_train)

        X_train = tokenizer.texts_to_sequences(review_train)
        # create "prod_test" dataframe which concatenates user input with X testing dataset
        prod_test = pd.concat([input_for_review_pred["new_desc"], review_test])
        prod_test = tokenizer.texts_to_sequences(prod_test)

        vocab_size = len(tokenizer.word_index) + 1  # reserve 0 empty index

        max_len =70

        X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
        prod_test = pad_sequences(prod_test, padding='post', maxlen=max_len)

        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        prod_test = ss.transform(prod_test)

        embedding_dim = 100

        model = Sequential()
        model.add(layers.Embedding(vocab_size, embedding_dim, input_length=max_len))
        model.add(layers.Conv1D(128, 5, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, kernel_initializer="normal"))

        # load exported model weights
        model.load_weights("model_wine.h5")

        model.compile(optimizer='adam',loss='mse',metrics=[metrics.mse])
        loss, acc = model.evaluate(X_train, y_train, verbose=False)
        
        # Return the prediction to the user
        st.write("Based on your review, the estimated quality of this wine is: ")
        val = model.predict(prod_test[0:1])[0][0]
        st.write(val)
        # Recall that the rest of the testing dataset was appended to the user input data
        # (The streamlit model is capabable of returning the calculations for the full testing dataset 
        # if this line of code (model.predict(prod_test[0:1])[0][0]) is changed to simply (prod_test)

def recommendation_demo():
    # instruct the user on how to use the application
    st.write("# Wine Recommendations! ")   
    st.write("# What qualities are most important for you?")
    st.write(
        """
        What kind of wines do you like? 
 
        Use the input box to describe what kind of wine you're looking for - taste, aroma, pairing, etc and we'll find the best match based on our inventory.                 
    """
    )

    # give the user a text box to enter a list of terms
    terms_list = st.text_input("Enter any text information that describes the kind of wine you're looking for, and we'll show you our best match!", value="fruity merlot cheese")

    # create a list from the data and capitalize the data for easy matching
    terms_list_upper = terms_list.split()
    terms_list_upper = [term.upper() for term in terms_list_upper]

    # use historical dataset (with text data) for matching
    df_reviews_searching = df_reviews.copy().drop(['points', 'price', 'id'], axis = 1)
    df_reviews_searching = df_reviews_searching.loc[:,'country':'winery'].apply(lambda x: x.str.upper())

    # assign a score to the best matches
    #https://stackoverflow.com/questions/64146240/python-searching-data-frame-for-words-in-a-list-and-keep-track-of-words-found-an

    # create additional columns to count the # of word matches with the terms the user input
    review_cols = df_reviews_searching.columns
    for col in review_cols:
        # name the new columns (which just appends _words and _count to the old column names)
        new_col_words = f"{col}_words"
        new_col_count = f"{col}_count"
        # calculate the number of times the search terms show up in each wine
        df_reviews_searching[new_col_words] = df_reviews_searching[col].str.findall('({0})'.format('|'.join(terms_list_upper)), flags=re.IGNORECASE)
        df_reviews_searching[new_col_count] = df_reviews_searching[new_col_words].str.len()

    # calculate the overall "score"
    df_reviews_searching["score"] = df_reviews_searching["description_count"] + df_reviews_searching["designation_count"] + df_reviews_searching["province_count"] + df_reviews_searching["region_1_count"] + df_reviews_searching["region_2_count"] + df_reviews_searching["taster_name_count"] + df_reviews_searching["taster_twitter_handle_count"] + df_reviews_searching["title_count"] + df_reviews_searching["variety_count"] + df_reviews_searching["winery_count"]

    # rank them 
    df_reviews_searching = df_reviews_searching.sort_values("score", ascending = False).head(1)

    # display the most highly ranked wine (based on search term matches) to the user
    st.write("Based on your preferences, we recommend this wine!")
    st.write(f"Title: {df_reviews_searching.iloc[0, 8]}")
    st.write(f"Variety: {df_reviews_searching.iloc[0, 9]}")   
    st.write(f"Designation: {df_reviews_searching.iloc[0, 2]}")
    st.write(f"Country of origin: {df_reviews_searching.iloc[0, 0]}")
    st.write(f"Province: {df_reviews_searching.iloc[0, 3]}")
    st.write(f"Region: {df_reviews_searching.iloc[0, 4]}, {df_reviews_searching.iloc[0, 5]}")
    st.write(f"Winery: {df_reviews_searching.iloc[0, 10]}")
    st.write(f"\nTaster name: {df_reviews_searching.iloc[0, 6]}")
    st.write(f"\nTaster twitter handle: {df_reviews_searching.iloc[0, 7]}")
    st.write(f"\nDescription: {df_reviews_searching.iloc[0, 1]}")
    
def stats_demo():
    #instruct the user on how to use the application
    st.write("# Wine by the Numbers")
    st.write("#### User our interactive dashboard to help you find the best wines that fit your tastes!")
    st.text(" \n")
    st.write("### Make a selection below to find the best wines associated with your preference:")
    
    #application displays data in a click-to-reveal method
    #if the preference most important to the user is "acidity" of a wine, then "Chardonnay" is displayed, etc
    #(These results were obtained from Gautam's NLP models)
    #documentation reference: https://discuss.streamlit.io/t/is-is-possible-i-place-a-button-or-check-box-at-any-positon-in-the-page/8032/2
    
    #create buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        but1 = st.button("acidity")
    with col2:
        but2 = st.button("fruity")
    with col3:
        but3 = st.button("aromas")
    with col4: 
        but4 = st.button("palate")
    with col5:
        but5 = st.button("tannins")

    #display results depending on which button the user clicks
    col1, col2, col3, col4, col5 = st.columns(5)
    if but1:
        with col1:
            st.write("Chardonnay")
    if but2:
        with col2:
            st.write("Pinot Noir")
    if but3:
        with col3:
            st.write("Red Blend")
    if but4:
        with col4:
            st.write("Riesling")
    if but5:
        with col5:
            st.write("Cabernet Sauvignon")

    
    # Display figures to help consumers understand the following:
    # 1) Top 10 Countries by Average Points
    # 2) Top 10 Varieties by Average Points
    # 3) Top 10 Wineries by Average Points
    # 4) Top 10 Countries by Average Points per Dollar
    st.text(" \n")
    st.write("### The Best Bang for your Buck!:")
    st.text(" \n")
    
    #
    # Figure creation code taken from Gautam's notebook X and adapated for use in the streamlit application
    # 

    # 1) Create figure for top 10 Countries by Average Points
    fig, ax = plt.subplots(figsize=(15,5))
    sns.barplot(x=df_reviews.groupby("country").mean().sort_values(by="points",ascending=False).price.index[:10], y=df_reviews.groupby("country").mean().sort_values(by="points",ascending=False).points.values[:10], palette="hls", ax=ax)#.set(title='Title of Plot')
    plt.ylabel("Average Points", fontsize=30)
    plt.title("Top 10 Countries by Average Points:", fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel(None)
    plt.xticks(rotation= 45, ha="right")
    for p in ax.patches:
        ax.annotate("%.1f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=17, color='gray', xytext=(0, 20),
                 textcoords='offset points')
    #
    ax.set_ylim([85,95])
    st.pyplot(fig)
    st.text(" \n")
    
    # 2) Top 10 Varieties by Average Points
    fig2, ax2 = plt.subplots(figsize=(20,5))
    sns.barplot(x=df_reviews.groupby("variety").mean().sort_values(by="points",ascending=False).price.index[:10], y=df_reviews.groupby("variety").mean().sort_values(by="points",ascending=False).points.values[:10], palette="hls", ax=ax2)#.set(title='Title of Plot')
    plt.ylabel("Average Points", fontsize=30)
    plt.title("Top 10 Varieties by Average Points:", fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel(None)
    plt.xticks(rotation= 45, ha="right")
    for p in ax2.patches:
        ax2.annotate("%.1f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=17, color='gray', xytext=(0, 20),
                 textcoords='offset points')
    ax2.set_ylim([90,98])
    st.pyplot(fig2)
    st.text(" \n")

    # 3) Top 10 Wineries by Average Points
    fig3, ax3 = plt.subplots(figsize=(20,5))
    sns.barplot(x=df_reviews.groupby("winery").mean().sort_values(by="points",ascending=False).price.index[:10], y=df_reviews.groupby("winery").mean().sort_values(by="points",ascending=False).points.values[:10], palette="hls", ax=ax3)#.set(title='Title of Plot')
    plt.ylabel("Average Points", fontsize=30)
    plt.title("Top 10 Wineries by Average Points:", fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel(None)
    plt.xticks(rotation= 45, ha="right")
    for p in ax3.patches:
        ax3.annotate("%.1f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=17, color='gray', xytext=(0, 20),
                 textcoords='offset points')
    ax3.set_ylim([90, 100])
    st.pyplot(fig3)
    st.text(" \n")

    #4) Top 10 Countries by Average Points per Dollar
    df_reviews_2 = df_reviews[np.isfinite(df_reviews["price"])]
    df_reviews_2["points/price"] = df_reviews_2.points / df_reviews_2.price
    df_reviews_2.groupby("country").mean().sort_values(by="points/price", ascending=False)
    fig4, ax4 = plt.subplots(figsize=(20,5))
    sns.barplot(x=df_reviews_2.groupby("country").mean().sort_values(by="points/price", ascending=False)["points/price"].index[:10], y=df_reviews_2.groupby("country").mean().sort_values(by="points/price", ascending=False)["points/price"].values[:10], palette="hls", ax=ax4)
    plt.ylabel("Average Points per Dollar", fontsize=30)
    plt.title("Top 10 Countries by Average Points per Dollar:", fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel(None)
    plt.xticks(rotation= 45, ha="right")
    for p in ax4.patches:
        ax4.annotate("%.1f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=17, color='gray', xytext=(0, 20),
                 textcoords='offset points')
    ax4.set_ylim([4, 12])
    st.pyplot(fig4)

def recommendation_demo_nlp():
    
    # Instruct the user on how to use the application
    st.write("# Advanced Wine Recommendations based on Sentiment Models")

    description = st.text_input("Enter a word that best describes what you're looking for in a wine:", value="fruity")
    sentiment_filter = st.number_input("For personalized recommendations, enter the minimum sentiment score you're looking for in a wine:", min_value = 0.0, max_value = 1.0, value = 0.9, step = 0.1)

    # Display word references
    st.write("Keep in mind some words may not have many matching results...")
    st.write("Here are the most popular words in our dataset:")
    image = Image.open('word_reference.png')
    st.image(image, caption='Most prevalent words in wine reviews dataset')
    # Photo from project presentation
    
    # if the user clicks "calculate", the model will return the top variety of wine that meets their sentiment score preferences
    # and will also return the best wine by sentiment score
    if st.button("Calculate!"):
        st.write("Calculating...")

        # Create sentiment scores dataset
        # Code used from Gautam's sentiment score notebook
        st.write("Importing required libraries...")
        from nltk.tokenize.treebank import TreebankWordDetokenizer
        from nltk.corpus import stopwords
        from nltk import word_tokenize
        nltk.download('vader_lexicon')

        # create sentiment scores dataset
        df_sentiment, df, data = df_reviews.copy(), df_reviews.copy(), df_reviews.copy() #match references with terms used in notebook
        sid = SentimentIntensityAnalyzer()
        df_sentiment.reset_index(inplace=True, drop=True)
        st.write("Applying polarity scores...")
        df_sentiment[['neg', 'neu', 'pos', 'compound']] = df_sentiment['description'].apply(sid.polarity_scores).apply(pd.Series)
        df_sentiment["variety"] = df["variety"]
        df_sentiment.dropna()

        #use TreebankWordDetokenizer()
        stopwords = set(stopwords.words('english'))
        detokenizer = TreebankWordDetokenizer()

        def clean_description(desc):
            desc = word_tokenize(desc.lower())
            desc = [token for token in desc if token not in stopwords and token.isalpha()]
            return detokenizer.detokenize(desc)

        st.write("Applying word tokenization...")
        data["cleaned_description"] = data["description"].apply(clean_description)
        word_occurrence = data["cleaned_description"].str.split(expand=True).stack().value_counts()
  
        # in Gautam's notebook, the description associated with df_fruit was fruity
        # it is substituted for "description" here but "df_fruit" term is kept the same for easy reference
        st.write("Creating personalized recommendations...")
        df_fruit = pd.DataFrame(data["description"].str.contains(description))
        df_fruit = df_fruit[df_fruit['description'] == True]

        df_fruit["description"] = data["description"]
        df_fruit["title"] = data["title"]
        df_fruit["Sentiment Score"] = df_sentiment["compound"]
        df_fruit["Variety"] = df_sentiment["variety"] 

        avg_sentiment_score = df_fruit["Sentiment Score"].mean()
        wine_count = len(df_fruit)

        st.write("Done!")
        st.write("\n")

        st.write(f"The total number of wines in our dataset associated with {description} are {wine_count}")
        st.write(f"The average sentiment score with wines in our dataset associated with {description} is {avg_sentiment_score:0.2f}")

        #sentiment_filter set to 0.9 by default
        filter1 = df_fruit["Sentiment Score"] > sentiment_filter    
        good_fruit = df_fruit.loc[filter1]
        good_fruit.dropna()

        # Display the top variety for the user
        best_variety = good_fruit["Variety"].describe()["top"]
        st.write(f"The top variety of wine associated with {description} is {best_variety}")
        
        # Display the top wine by sentiment score for the user
        good_fruit = good_fruit.sort_values("Sentiment Score", ascending = False)
        
        good_fruit_display = good_fruit[["Sentiment Score", "title", "Variety", "description"]].copy()
        st.write("\n")
        st.write("Our top 5 recommendations based on sentiment score are:")
        st.dataframe(good_fruit_display.head())
    
# Dictionary for sidebar
page_names_to_funcs = {
    "Welcome!": intro,
    "Wine Quality Calculator": quality_demo,
    "Wine Quality Predictor": reviews_demo,
    "Wine Simple Recommendation Page": recommendation_demo,
    "Wine Advanced Recommendation Page": recommendation_demo_nlp,
    "Wine by the Numbers": stats_demo,
}

# sidebar selector
demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()