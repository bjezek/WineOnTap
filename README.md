# WineOnTap

As an individual, it might be easier to try a new wine, even without a review. But as a business, can you take that chance? Something new can be a competitive advantage for you, but could still be a risk if it's a flop.

After reviewing wine datasets available online, the project team used knowledge that they acquired from the FinTech Bootcamp program at Rice University to create various models that would help people make better decisions about wine and their purchases. The models have been integrated into a user-friendly streamlit application that can be deployed on localhost.  

The work in this repository is organzed by team member name, the following list denotes each team member's responsibilities for the project: 
* Brandon - random forest model
* Gautam - natural language processing model
* Jerome - neural network model
* Paula - streamlit application development

# Installation instructions

Use `git clone` to download the project repository to download the associated files. 

The models were developed using a selection of jupyter notebook files that are included in this repository. 

The files located in the streamlit application folder (see Paula's folder) are enough to run the streamlit application. The environment needed to run the streamlit application can be created by creating a new environment using Python version 3.7.13 and installing tensorflow version 2.9.2. 

To run the streamlit applicaiton, `cd` into the streamlit application folder and type `streamlit run wine_ml.py`. A browser should open with the application deployed on localhost. 

# Resources

The following files were used to develop the application models, which are readily available online: 

* [Chemical composition data](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)
    * Sources downloaded from this location for `winequality-red.csv` and `winequality-white.csv`
* [Wine reviews data](https://www.kaggle.com/datasets/mysarahmadbhat/wine-tasting)
    * Sources downloaded from this location for `winemag-data-130k-v2.csv`
* [Wine reviews data](https://www.kaggle.com/datasets/zynicide/wine-reviews)
    * Sources downloaded from this location for `wine_mag-data_first-150k.csv`

Citations for work referenced in neural network model development are located in Jerome's folder.

# Modeling

Random forest models, natural language processing models, and neural network models were created for this project, see more detailed descriptions below about the models. 

## Random Forest Model For predicting 
The random forest model was created to predict the rating of a wine
based on important factors that made a red or white wine the best in terms of quality.

(See Brandon's folder for jupyter notebook files.) 

### Heat Map showing Correlation 
![image](https://user-images.githubusercontent.com/106267420/202585106-e78c48f7-b747-47c4-9883-0b6d0b6415fe.png)
![image](https://user-images.githubusercontent.com/106267420/202585414-2a79c9c0-a989-4006-8c0c-332dae89ecc7.png)

### Feature importance ! (Red wine First, White Wine Second displayed)
![image](https://user-images.githubusercontent.com/106267420/202585953-d743f81e-f808-4531-97cb-da5a06f8327b.png)
![image](https://user-images.githubusercontent.com/106267420/202586455-96f38142-58f3-4dda-9609-55aef970238f.png)

## Sentiment Analysis 

Sentiment Analysis was applied on the wine reviews dataset to determine the best type of wine that is associated with personal preferences of a specific attribute (acidity, aroma, etc) 

For example, if you like a fruity wine, based on tasting reviews, sentiment scores, and filtering, a pinot noir wine likely suits your tastes.

(See Gautam's folder for jupyter notebook files.) 


<img width="1062" alt="Screen Shot 2022-11-15 at 7 56 24 PM" src="https://user-images.githubusercontent.com/107082333/202930328-b1961369-1ad6-4769-bae7-a424d8293459.png">

![Image 11-15-22 at 7 32 PM](https://user-images.githubusercontent.com/107082333/202930391-3215e7fc-b26a-473c-856f-05960b4e2507.jpg)

![Image 11-15-22 at 7 32 PM](https://user-images.githubusercontent.com/107082333/202930382-c09c51e7-faa8-4dc3-958b-063bd0b61b5f.jpg)

## Neural Network Models

Neural network models were applied to the wine tasting reviews dataset to predict the quality of a wine based on a review for that wine. 

(See Jerome's folder for associated files)

# Streamlit Application for Wine Recommendation System

A Streamlit application was created that combined the results of these models into a user-friendly application.

The video in the root directory of the project titled `streamlit-app-screen-recording.wbm` shows how the application can be used and shows the models in action. 

## Model Integration

The models created during application development were integrated into the Streamlit application as follows: 
* Random Forest Model 
    * Model was exported as a pickle file and imported into the notebook
    * Model predicts the quality of a wine based on chemical composition
    * See the model in action! Check out this section of the video: 0:11 - 0:41
* Natural Language Processing Model
    * Model results were incorported into the application in the form of buttons to help the user determine which wines they should consider based on their prefereces
    * Model results were incorporated into the application to match user inputted preferences based on the sentiment of the tasting reviews data
    * Model recommends varieties of wine and wines themselves based on the reviews with the highest sentiment
    * See the model in action! Check out this section of the video: 2:56 - 5:30
* Neural Network Model
    * Model was exported as a .h5 file and imported into the notebook
    * Model predicts the estimated quality of a wine based on a user review
    * See the model in action! Check out this sectino of the video: 0:41 - 2:30

## User Input Capture & Display of Model Results

User input was captured mainly using `st.number_input` or `st.text_input`. That information was then used to create a dataframe (similar in structure to original model development) that was then fed to the model for prediction purposes. To honor the way the models were created, the data was scaled with the testing data. The first row of the dataset fed to the models for prediction purposes contains the user-inputted data, which is then returned within the streamlit application to the user.   

One potential optimization is to export the fitted `standardscaler()` model and import that into the notebook. Due to issues experienced during application development the standardscaler() model is re-created within the streamlit application (as it was created in the original workbooks done by project team members).

## Video highlights

See the models in action! Check out the following time ranges in the video to see each model in action: 

Random forest (Brandon): 0:11 - 0:41

Natural language processing (Gautam): 2:56 - 5:30

Neural Network Model (Jerome): 0:41 - 2:30

(Keep in mind it takes a few minutes to run the neural network models and natural language processing models so please be patient.)






