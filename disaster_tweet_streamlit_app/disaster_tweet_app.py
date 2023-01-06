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
import joblib

if 'vectorizer_glove' not in st.session_state:
    st.session_state['vectorizer_glove'] = gensim.downloader.load('glove-twitter-200')
    st.write("vectorizer_glove downloaded....")

eda = st.container()
data_prep_machine_learning = st.container()
user_input = st.container()

#setting random state to 1
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
        'Which vectoriser do you want to use', ('Glove', 'Word2vec', 'FastText'))

    # asking user for model selectiom
    model_selection_ouput = form_dp.selectbox(
        'Which model do you want to select ?', ('Random Forest Classifier', 'XGBClassifier', ))

    # asking user for number of estimators
    estimators_input = form_dp.slider(
        'What should be the number of trees?', min_value=100, max_value=600, step=100)

    # #asking user for number of folds
    number_of_folds_input = form_dp.slider(
        'How many folds in Cross Validation?', min_value=2, max_value=10, step=1)

    data_prep_form_submit_button_output = form_dp.form_submit_button(
        "Submit for data preparations")

    if data_prep_form_submit_button_output:
        if data_cleaning_ouput == 'Yes':
            train_df = util.data_cleaning(tweets_df)

        # vectorisation
        # if vectoriser_output == 'CountVectoriser':
        #     cv = CountVectorizer()
        #     train_df, vectorizer = util.vectorization_df(cv, train_df)

        # if vectoriser_output == 'TfidVectoriser':
        #     tf = TfidfVectorizer()
        #     train_df, vectorizer = util.vectorization_df(tf, train_df)

        if vectoriser_output == 'Glove':
            # initializing glove vectorizer
            vectorizer = st.session_state['vectorizer_glove']
            # vectorizer = gensim.downloader.load('glove-twitter-200')

            # storing vector of tweet in tweet vector
            train_df['tweet_vector'] = train_df['cleaned_text'].apply(
                lambda x: util.tweet_vec(x, vectorizer))

            # dropping na values produced after vectorisation in tweet_vector column
            train_df.dropna(subset=['tweet_vector'], inplace=True)

            # taking average of the vectors in a tweet.example: if a tweet has 3 words,
            #  there will be 3 vectors of 200 dimensions each
            # i will take average of all the vectors to get one vector of 200 dimensions
            train_df['average_vector'] = train_df['tweet_vector'].apply(
                util.average_vec)

        if vectoriser_output == 'Word2vec':


            vectorizer = gensim.downloader.load('word2vec-google-news-300')
            train_df['tweet_vector'] = train_df['cleaned_text'].apply(
                lambda x: util.tweet_vec(x, vectorizer))
            train_df.dropna(subset=['tweet_vector'], inplace=True)
            train_df['average_vector'] = train_df['tweet_vector'].apply(
                util.average_vec)

        if vectoriser_output == 'FastText':
            vectorizer = gensim.downloader.load('word2vec-google-news-300')
            train_df['tweet_vector'] = train_df['cleaned_text'].apply(
                lambda x: util.tweet_vec(x, vectorizer))
            train_df.dropna(subset=['tweet_vector'], inplace=True)
            train_df['average_vector'] = train_df['tweet_vector'].apply(
                util.average_vec)

        if model_selection_ouput == 'Random Forest Classifier':
            model_selection = RandomForestClassifier(
                n_estimators=estimators_input, random_state=RANDOM_STATE, n_jobs=-1)
        else:
            model_selection = XGBClassifier(n_estimators=estimators_input,
                                            random_state=RANDOM_STATE, n_jobs=-1)

        # #for tfidf and count vectorizers, the training, predicting procedure
        # if (vectoriser_output == 'TfidVectoriser' or vectoriser_output == 'CountVectoriser'):
        #     st.write(train_df.head(1))

        #     #training the model on whole dataset, to save the model for user input predictions
        #     fitted_model_to_save = model_selection.fit(train_df, tweets_df['target'] )

        #     #adding the label class since it is required by my function
        #     train_df["target"] = tweets_df["target"]
        #     X_train, X_test, y_train, y_test = util.ttsplit(train_df)

        #     result_dic = util.training_eval(
        #         model_selection, X_train, X_test, y_train, y_test)

        #     st.write('The f1 score is:', result_dic['f1'])
        #     st.write('The precision score is:', result_dic['precision'])
        #     st.write('The recall score is:', result_dic['recall'])
        #     st.write('The roc auc  is:', result_dic['roc'])

        # for pretrained vectorizers, cross validation,

        if (vectoriser_output == 'Glove' or vectoriser_output == 'Word2vec' or vectoriser_output == 'FastText'):

            # stacking the average vectors in array[[]] format, else  format was arr[arr[],arr[],arr[]],
            #  stacking the whole dataset, so that whole dataset can be trained .

            train_set = np.vstack(train_df['average_vector'])

            # training the model on whole dataset, to save the model for later use
            fitted_model_to_save = model_selection.fit(
                train_set, train_df['target'])

            st.write(train_df.head(1))
            output_dic = util.cv_score_model(df=train_df[[
                                             'target', 'average_vector']], folds = number_of_folds_input,
                                             model=model_selection, feature_column='average_vector')

            st.write('Mean F1 score is:   ', output_dic['f1'])
            st.write('Mean precision score is: ', output_dic['precision'])
            st.write('Mean recall score is: ', output_dic['recall'])
            st.write('Mean roc score is: ', output_dic['roc'])

        # saving current direction in current_dir
        current_dir = os.path.dirname(__file__)

        # making a path way for the model to be saved, this folder is git -ignored ,
        #  since i dont want it to be pushed to github everytime
        #
        saved_model = os.path.join(current_dir, '.', 'saved_data/saved_model')
        # saving the trained model
        joblib.dump(fitted_model_to_save, saved_model)
        # saving the pretrained vectorizer
        saved_vectorizer = os.path.join(current_dir, '.', 'saved_data/saved_vectorizer')
        joblib.dump(vectorizer, saved_vectorizer)

        # st.write(train_df.head(1))



with user_input:

    st.header('Classifying your tweet input ')

    # initializing a form to input user input
    input_form = st.form(key='user')

    user_input_tweet = input_form.text_input(
        'Enter a tweet to check if it is a disaster related tweet or not')
    st.write('Your input:', user_input_tweet)

    # when user presses the submit button, it will be stored in this
    user_form_submit_button = input_form.form_submit_button(
        'Submit your input')

    # after user presses the submit button this  then the following steps will execute
    if user_form_submit_button:
        # cleaning user input text
        cleaned_text = util.user_input_data_cleaning(user_input_tweet)

        # setting current directory
        current_dir = os.path.dirname(__file__)
        saved_model = os.path.join(current_dir, '.', 'saved_data/saved_model')

        # loading the saved model from the disk
        model_load = joblib.load(saved_model)

        # loading saved vectorizer from disk
        saved_vectorizer = os.path.join(current_dir, '.', 'saved_data/saved_vectorizer')
        vectorizer_load = joblib.load(saved_vectorizer)

        #count and tfidf
        # output = vectorizer_load.transform([cleaned_text])

        # pre trained vectorizer
        vectorized_input_tweet = util.tweet_vec(cleaned_text, vectorizer_load)

        vectorized_input_tweet_average = util.average_vec(
            vectorized_input_tweet)
        vectorized_input_tweet_average = np.reshape(
            vectorized_input_tweet_average, (1, -1))

        pred = model_load.predict(vectorized_input_tweet_average)
        if (pred == 0):
            st.write('Output: It is not a disaster related tweet')
        else:
            st.write('Output: It is a disaster related tweet')
