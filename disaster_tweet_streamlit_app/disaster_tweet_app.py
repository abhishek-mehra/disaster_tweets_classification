from matplotlib.style import use
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from xgboost import XGBClassifier
import os
import gensim
import gensim.downloader
import util

eda = st.container()
data_prep_machine_learning = st.container()
user_input = st.container()
metrics = st.container()

RANDOM_STATE = 1

train_df = pd.DataFrame()

# sets the path of current directory
current_dir = os.path.dirname(__file__)
# this will tell that file is in previous directory i.e one step behind the current directory in data folder
train_data_fp = os.path.join(current_dir, "..", "data/train.csv")


with eda:
    st.header("Disaster Tweet Classification- Data Science Project")
    st.header('Exploratory Data Analysis')

    # importing the dataset

    tweets_df = pd.read_csv(train_data_fp)

    st.write(tweets_df.tail(5))

    # most occuring locations
    # location_button_output = st.button(
        # 'Click to see the what are the  most occuring locations in the tweets')
    # if location_button_output:
    fig = util.location_occurence(tweets_df)
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


with data_prep_machine_learning:
    st.header('Machine Learning')

    # making a form for data preparations options
    form_dp = st.form(key='dp')

    # asking for data cleaning options
    data_cleaning_ouput = form_dp.selectbox(
        'Do you want to clean the tweets off html tags, smileys, URLs', ('Yes', 'No'))

    # asking for vectorization method
    vectoriser_output = form_dp.selectbox(
            'Which vectoriser do you want to use', ('CountVectoriser', 'TfidVectoriser', 'Glove', 'Word2vec', 'FastText'))

    model_selection_ouput = form_dp.selectbox(
        'Which model do you want to select ?', ('Random Forest Classifier', 'XGBClassifier', ))

    estimators_input = form_dp.slider(
        'What should be the number of trees?', min_value=100, max_value=600, step=100)

    data_prep_form_submit_button_output = form_dp.form_submit_button(
        "Submit for data preparations")

    if data_prep_form_submit_button_output:
        if data_cleaning_ouput == 'Yes':
            train_df = util.data_cleaning(tweets_df)

        # vectorisation
        if vectoriser_output == 'CountVectoriser':
            cv = CountVectorizer()
            train_df = util.vectorization_df(cv, train_df)

        if vectoriser_output == 'TfidVectoriser':
            tf = TfidfVectorizer()
            train_df = util.vectorization_df(tf, train_df)

        if vectoriser_output == 'Glove':
            glove_twitter = gensim.downloader.load('glove-twitter-200')
            train_df['tweet_vector'] = train_df['cleaned_text'].apply(lambda x: util.tweet_vec(x, glove_twitter))
            train_df.dropna(subset=['tweet_vector'], inplace=True)
            train_df['average_vector'] = train_df['tweet_vector'].apply(
                util.average_vec)

        if vectoriser_output == 'Word2vec':
            word_2_vec = gensim.downloader.load('word2vec-google-news-300')
            train_df['tweet_vector'] = train_df['cleaned_text'].apply(lambda x: util.tweet_vec(x, word_2_vec))
            train_df.dropna(subset=['tweet_vector'], inplace=True)
            train_df['average_vector'] = train_df['tweet_vector'].apply(
                util.average_vec)


        if vectoriser_output == 'FastText':
            fast_t = gensim.downloader.load('word2vec-google-news-300')
            train_df['tweet_vector'] = train_df['cleaned_text'].apply(lambda x: util.tweet_vec(x, fast_t))
            train_df.dropna(subset=['tweet_vector'], inplace=True)
            train_df['average_vector'] = train_df['tweet_vector'].apply(
                util.average_vec)





        if model_selection_ouput == 'Random Forest Classifier':
            model_selection = RandomForestClassifier(
                n_estimators=estimators_input, random_state=RANDOM_STATE, n_jobs=-1)
        else:
            model_selection = XGBClassifier(n_estimators=estimators_input,
                                random_state=RANDOM_STATE, n_jobs=-1)

        if (vectoriser_output == 'TfidVectoriser' or vectoriser_output =='CountVectoriser'):
            st.write(train_df.head(1))
            train_df["target"] = tweets_df["target"]
            X_train, X_test, y_train, y_test = util.ttsplit(train_df)

            result_dic = util.training_eval(
                model_selection, X_train, X_test, y_train, y_test)

            st.write('The f1 score is:', result_dic['f1'])
            st.write('The precision score is:', result_dic['precision'])
            st.write('The recall score is:', result_dic['recall'])
            st.write('The roc auc  is:', result_dic['roc'])

        if (vectoriser_output == 'Glove' or vectoriser_output=='Word2vec' or vectoriser_output=='FastText') :
            st.write(train_df.head(1))
            output_dic = util.cv_score_model(df = train_df[['target','average_vector']], model=model_selection, feature_column='average_vector')

            st.write('Mean F1 score is:   ', output_dic['f1'])
            st.write('Mean precision score is: ', output_dic['precision'])
            st.write('Mean recall score is: ', output_dic['recall'])
            st.write('Mean roc score is: ', output_dic['roc'])



        # st.write(train_df.head(1))




with user_input:
    st.header('Input a tweet which you want to classify')



    form_ml = st.form(key='ml')
    # 2. asking user for Model selection: (a) Random Forest
    model_selection_ouput = form_ml.selectbox(
        'Which model do you want to select ?', ('Random Forest Classifier','XGBClassifier', ))

    # 3.asking user for Model selection: (a)number of estimators
    estimators_input = form_ml.slider(
        'What should be the number of trees?', min_value=100, max_value=600, step=100)

    # 3 asking user for max depth
    # max_depth_input = form_ml.slider(
        # 'What should be the max depth of trees?', min_value=2, max_value=8, step=1)

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
                                   random_state=RANDOM_STATE, n_jobs=-1)


    # splitting data
        train_df["target"] = tweets_df["target"]
        X_train, X_test, y_train, y_test = util.ttsplit(train_df)

        result_test_dic = util.training_eval(model, X_train, X_test, y_train, y_test)

        st.write('the f1 score is:', result_test_dic['f1'])
