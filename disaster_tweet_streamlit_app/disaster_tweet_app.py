from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from xgboost import XGBClassifier
import os

import util

eda = st.container()
data_prep = st.container()
machine_learning = st.container()
metrics = st.container()

RANDOM_STATE = 1

train_df = pd.DataFrame()

current_dir = os.path.dirname(__file__)
train_data_fp = os.path.join(current_dir, "..", "data/train.csv")


with eda:
    st.header("Disaster Tweet Classification- Data Science Project")
    st.header('Exploratory Data Analysis')

    # importing the dataset

    tweets_df_train = pd.read_csv(train_data_fp)

    st.write(tweets_df_train.tail(5))

    # most occuring locations
    # location_button_output = st.button(
        # 'Click to see the what are the  most occuring locations in the tweets')
    # if location_button_output:
    fig = util.location_occurence(tweets_df_train)
    st.pyplot(fig)

    # # most occuring keywords in tweets
    # keyword_button__output = st.button(
    #     'Click to see what are the  most occuring keyword in tweet')
    # if keyword_button__output:
    #     fig = util.keyword_occurence(tweets_df_train)
    #     st.pyplot(fig)

    # # number of words in a tweet
    # words_button_ouput = st.button(
    #     'Click to see the number of words distribution in the tweets')

    # # most common words in a tweet
    # common_words_button_ouput = st.button(
    #     'Click to see most common words in the tweet')


with data_prep:
    st.header('Data Preparations For Machine Learning')
    
    form_dp = st.form(key='dp')

    data_cleaning_ouput = form_dp.selectbox(
        'Do you want to clean the tweets off html tags, smileys, URLs',('Yes','No'))

    vectoriser_output = form_dp.selectbox(
            'Which vectoriser do you want to use', ('CountVectoriser', 'TfidVectoriser'))


    data_prep_form_submit_button_output = form_dp.form_submit_button(
        "Submit for data preparations")

    
    if data_prep_form_submit_button_output:
        if data_cleaning_ouput == 'Yes': 
            tweets_df_train = util.clean_tweets(tweets_df_train, "text")

        #filling missing values in keyword and location
        tweets_df_train = util.fill_na(tweets_df_train)

        #trim unwanted spaces
        tweets_df_train = util.trim_spaces(tweets_df_train)

        # vectorisation
        

        if vectoriser_output == 'CountVectoriser':
            train_df = util.clean_vectorize_using_count_vectorizer(tweets_df_train,'text')

        else:
            train_df = util.clean_vectorize_using_tfidf_vectorizer(tweets_df_train,'text')

        #getting dummies for location and keyword

        gd_keyword, gd_location = util.make_dummies(tweets_df_train)

        train_df = pd.concat([train_df, gd_keyword, gd_location],axis=1)  #joining vectorised text,location and keyword features


    
        train_df["target"] = tweets_df_train["target"]

        st.write(train_df.head())
        st.write(train_df.shape)
        st.write(train_df['target'])

    


with machine_learning:
    st.header('Training a model to classify tweets')

    

    form_ml = st.form(key='ml')
    # 2. asking user for Model selection: (a) Random Forest
    model_selection_ouput = form_ml.selectbox(
        'Which model do you want to select ?', ('Random Forest Classifier','XGBClassifier', ))

    # 3.asking user for Model selection: (a)number of estimators
    estimators_input = form_ml.slider(
        'What should be the number of trees?', min_value=100, max_value=600, step=100)

    # # 3 asking user for max depth
    # max_depth_input = form_ml.slider(
    #     'What should be the max depth of trees?', min_value=2, max_value=8, step=1)

    # 4 asking user for cv folds
    # n_folds = form_ml.slider('How many CV folds?',
                            #  min_value=5, max_value=10, step=1)

    ml_form_submit_button_output = form_ml.form_submit_button(
        "Submit for training and evaluation")


    if ml_form_submit_button_output:
        if model_selection_ouput == 'Random Forest Classifier':
            model = RandomForestClassifier(
                n_estimators=estimators_input, random_state=RANDOM_STATE, n_jobs=-1)
        else:
            model = XGBClassifier(n_estimators=estimators_input,
                                  max_depth=max_depth_input, random_state=RANDOM_STATE, n_jobs=-1)

        # splitting data
        X_train, X_test, y_train, y_test = util.ttsplit(train_df)
        result_test_dic = util.training_eval(model, X_train, X_test, y_train, y_test)
        st.write('the f1 score is:', result_test_dic['f1'])
