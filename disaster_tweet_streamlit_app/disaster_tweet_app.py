from imp import load_compiled
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from xgboost import XGBClassifier, train
import os

import util
from sentence_transformers import SentenceTransformer
from gensim.models import KeyedVectors


current_dir = os.path.dirname(__file__)

# storing the vectorizer in the session state, for reruns it will not load up again and again
# if 'vectorizer_glove' not in st.session_state:
#     glove_path = os.path.join(current_dir, "..", "model/glove_vectorizer")
#     st.session_state['vectorizer_glove'] = loaded = KeyedVectors.load(glove_path)
#     st.write("vectorizer_glove downloaded....")


eda = st.container()
data_prep_machine_learning = st.container()
user_input = st.container()

# setting random state to 1
RANDOM_STATE = 1

# initializing sessions
if 'clf_model' not in st.session_state:
    st.session_state['clf_model'] = 'fitted_model'

if 'option' not in st.session_state:
    st.session_state['option'] = 'countfid'

if 'vectorizer' not in st.session_state:
    st.session_state['vectorizer'] = 'vectorizer'

train_df = pd.DataFrame()

# sets the path of current directory
current_dir = os.path.dirname(__file__)
# this will tell that file is in previous directory i.e one step behind the current directory in data folder
train_data_fp = os.path.join(current_dir, "..", "data/train.csv")


with eda:
    st.header("Disaster Tweet Identifier ")

    st.markdown(
            """Twitter is a platform for news and public discourse.
            It has 237 million regular users who share news, opinions, and topics.
            It also serves as a location for people to share incidents and disasters.
            Many organizations are interested in monitoring Twitter for emergency updates due to its real-time nature.""")


    eda.markdown("""This application enables you to select parameters to construct a machine
    learning model, and also allows you to input a tweet to determine if it pertains to a disaster.""")


    tweets_df = pd.read_csv(train_data_fp)

    with eda.expander("Know more about the app and the dataset"):

        st.markdown("""
        The first phase of the app allows users to train a model by selecting various
        parameters such as type of model, vectorizers, and relevant hyperparameters.
        In the second phase, users can input text, and the trained model from phase one can then
        be used to classify the input text.


        """)

        st.markdown(
        """
        About the dataset
        - The dataset has text, location , and target columns.
        - The text columns represents the tweets.
        - A keyword from that tweet.
        - The location the tweet was sent from.
        - In the target column 0 means not related to a disaster, while 1 means related to a disaster.
        """
            )

        st.write("Link to the dataset: https://www.kaggle.com/competitions/nlp-getting-started")


    # st.write("The dataset is picked from kaggle. It has 7613 tweets. We will train our Machine learning models on this dataset")
    st.subheader("Dataset")

    with eda.expander("Click to see the dataset"):
        st.dataframe(tweets_df, width = 1000)


    # most occuring locations
    # location_button_output = st.button(
    # 'Click to see the what are the  most occuring locations in the tweets')
    # if location_button_output:
    # fig = util.location_occurence(tweets_df)
    # st.pyplot(fig)

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


with eda.expander("Know more of the options to be utilized"):
    #     st.write("read more")

    st.markdown(
        """
        - The data cleaning button removes unnecessary information such as URLs, emojis from tweets, and also simplifies words to their base form (lemmatization) to improve the accuracy and efficiency of the model. """)

    st.markdown(
        """
        - Vectorizer will convert the tweets in vector(numerical form) for the machine learning model to understand.""")

    st.markdown(
        """
    - I have used 4 type of vectorizers:

        -  CountVectorizer- Convert a collection of text documents to a matrix of token counts.

        - TFIDF- Convert a collection of raw documents to a matrix of TF-IDF features.
         This vectorizer produces vectors  giving importance to word frequency in a document
        and more importance to less frequent words in collection of documents.

        - Glove- GloVe is an algorithm that generates numerical representations for words by analyzing their co-occurrence statistics in a corpus.
        It creates vector representations that capture the linear relationship between words in the vector space.

        Glove uses unsupervised learning to obtain vector representation of words.
        I have used glove trained on Gigaword dataset. It has 100 dimensional representation for a word.

        - ROBERTa - Sentence transformer based on roberta. This model has 768 dimensional representation for word.

    - Number of estimators - The number of trees in the forest.
    More help for model to learn from data, but more trees can also lead to overfitting.

    - Number of folds - is number of groups the dataset will be split into.
    The model will be trained on n-1 of folds and tested on remaining fold.
    This helps model to learn on all the data points.


        """)



with data_prep_machine_learning:
    st.subheader('Machine Learning training and cross-validation ')

    # making a form for data preparations options
    form_dp = st.form(key='dp')

    # asking for data cleaning options
    data_cleaning_ouput = form_dp.selectbox(
        'Do you want to perform data cleaning operations on the tweets.', ('Yes', 'No'))

    # asking for vectorization method
    vectoriser_output = form_dp.selectbox('Which vectoriser do you want to use',
                                          ('Glove_gigaword_100', 'Glove_twitter_50', 'TfidVectorizer', 'CountVectorizer', 'Distilroberta_v1'))

    # asking user for model selectiom
    model_selection_ouput = form_dp.selectbox(
        'Which model do you want to select ?', ('Random Forest Classifier', 'XGBClassifier', ))

    # asking user for number of estimators
    estimators_input = form_dp.slider(
        'What should be the number of trees?', min_value=100, max_value=500, step=100)

    # #asking user for number of folds
    number_of_folds_input = form_dp.slider(
        'How many folds in Cross Validation?', min_value=2, max_value=6, step=1)

    data_prep_form_submit_button_output = form_dp.form_submit_button(
        "Submit for machine learning training and cross-validation")

    if data_prep_form_submit_button_output:
        if data_cleaning_ouput == 'Yes':
            train_df = util.data_cleaning(tweets_df)

        # with st.expander("See explanation"):
        #     st.write(" The chart above shows some numbers I picked for you. I rolled actual dice for these, so they're *guaranteed* to be random.")

        if data_cleaning_ouput == 'No':
            train_df = tweets_df.rename(columns={'text': 'cleaned_text'})

        # vectorisation
        if vectoriser_output == 'CountVectorizer':
            path_count_vec = os.path.join(
                current_dir, "..", "model/count_vectorizer/count_vectorizer.pickle")
            vectorizer = util.load_pickle(path_count_vec)

            # train df just contains the count vectorised features
            train_df = util.vectorization_df(vectorizer, train_df)

        if vectoriser_output == 'TfidVectorizer':
            path_tfidf_vec = os.path.join(
                current_dir, "..", "model/Tfidf_vectorizer/tfidf_vectorizer.pickle")
            vectorizer = util.load_pickle(path_tfidf_vec)

            train_df = util.vectorization_df(vectorizer, train_df)

        if vectoriser_output == 'Distilroberta_v1':

            bert_path = os.path.join(
                current_dir, '..', 'model/distilroberta_v1')
            vectorizer = SentenceTransformer(bert_path)

            # either vectorise the dataset
            # distil_vectors = vectorizer.encode(train_df['cleaned_text'])
            # train_df['final_vector']  = pd.Series(list(distil_vectors))

            # load a pre vectorized dataset
            path_vectorized_df = os.path.join(
                current_dir, '..', 'model/distilroberta_v1/bert_vectorized_state.pickle')
            train_df = util.load_pickle(path_vectorized_df)

        if vectoriser_output == 'Glove_gigaword_100':
            # initializing glove vectorizer from the storage, which i have saved
            glove_wiki = os.path.join(
                current_dir, "..", "model/glove_gigaword_100/glove_wiki_gigaword_100.model")
            vectorizer = loaded = KeyedVectors.load(glove_wiki)
            # vectorizer = gensim.downloader.load('glove-twitter-200')

            # storing vector of tweet in tweet vector

            train_df['tweet_vector'] = train_df['cleaned_text'].apply(
                lambda x: util.tweet_vec(x, vectorizer))

            # dropping na values produced after vectorisation in tweet_vector column
            train_df.dropna(subset=['tweet_vector'], inplace=True)

            # taking average of the vectors in a tweet.example: if a tweet has 3 words,
            #  there will be 3 vectors of 200 dimensions each
            # i will take average of all the vectors to get one vector of 200 dimensions
            train_df['final_vector'] = train_df['tweet_vector'].apply(
                util.average_vec)

        if vectoriser_output == 'Glove_twitter_50':

            # vectorizer = gensim.downloader.load('word2vec-google-news-100')
            glove_twitter_50_path = os.path.join(
                current_dir, "..", "model/glove_twitter_50/glove-twitter-50.model")
            vectorizer = loaded = KeyedVectors.load(glove_twitter_50_path)

            train_df['tweet_vector'] = train_df['cleaned_text'].apply(
                lambda x: util.tweet_vec(x, vectorizer))
            train_df.dropna(subset=['tweet_vector'], inplace=True)
            train_df['final_vector'] = train_df['tweet_vector'].apply(
                util.average_vec)

        if model_selection_ouput == 'Random Forest Classifier':
            model_selection = RandomForestClassifier(
                n_estimators=estimators_input, random_state=RANDOM_STATE, n_jobs=-1)
        else:
            model_selection = XGBClassifier(n_estimators=estimators_input,
                                            random_state=RANDOM_STATE, n_jobs=-1)

        # for pretrained vectorizers-- cross validation,

        if (vectoriser_output == 'Glove_gigaword_100' or vectoriser_output == 'Glove_twitter_50' or vectoriser_output == 'Distilroberta_v1'):



            output_dic = util.cv_score_model(df=train_df[[
                'target', 'final_vector']], folds=number_of_folds_input,
                model=model_selection, feature_column='final_vector')

            st.write('Mean F1 score is:   ', output_dic['f1'])
            st.write('Mean precision score is: ', output_dic['precision'])
            st.write('Mean recall score is: ', output_dic['recall'])
            st.write('Mean roc score is: ', output_dic['roc'])

            # stacking the average vectors in array[[]] format, else  format was arr[arr[],arr[],arr[]],
            #  stacking the whole dataset, so that whole dataset can be trained .

            train_set = np.vstack(train_df['final_vector'])

            # training the model on whole dataset, to save the model for later use
            fitted_model = model_selection.fit(
                train_set, train_df['target'])

            if vectoriser_output == 'Distilroberta_v1':
                st.session_state['option'] = 'bert'
            else:
                st.session_state['option'] = 'pretrained'

            st.session_state['clf_model'] = fitted_model
            st.session_state['vectorizer'] = vectorizer

        if (vectoriser_output == 'CountVectoriser' or vectoriser_output == 'TfidVectoriser'):

            train_df['target'] = tweets_df['target']
            output_dic = util.cv_score_model(model=model_selection, df=train_df,
                                             folds=number_of_folds_input)

            st.write('Mean F1 score is:   ', output_dic['f1'])
            st.write('Mean precision score is: ', output_dic['precision'])
            st.write('Mean recall score is: ', output_dic['recall'])
            st.write('Mean roc score is: ', output_dic['roc'])

            fitted_model = model_selection.fit(
                train_df.drop('target', axis=1), train_df['target'])

            st.session_state['option'] = 'countfid'
            st.session_state['clf_model'] = fitted_model
            st.session_state['vectorizer'] = vectorizer

        # # saving current direction in current_dir
        # current_dir = os.path.dirname(__file__)

        # # making a path way for the model to be saved, this folder is git -ignored ,
        # #  since i dont want it to be pushed to github everytime
        # #
        # saved_model = os.path.join(current_dir, '.', 'saved_data/saved_model')
        # # saving the trained model
        # joblib.dump(fitted_model_to_save, saved_model)
        # # saving the pretrained vectorizer
        # saved_vectorizer = os.path.join(current_dir, '.', 'saved_data/saved_vectorizer')
        # joblib.dump(vectorizer, saved_vectorizer)

        # for tfidf and count vectorizers, the training, predicting procedure
        # if (vectoriser_output == 'TfidVectoriser' or vectoriser_output == 'CountVectoriser'):
        #     st.write(train_df.head(1))

        #     # training the model on whole dataset, to save the model for user input predictions
        #     fitted_model_to_save=model_selection.fit(
        #         train_df, tweets_df['target'])

        #     # adding the label class since it is required by my function
        #     train_df["target"]=tweets_df["target"]
        #     X_train, X_test, y_train, y_test=util.ttsplit(train_df)

        #     result_dic=util.training_eval(
        #         model_selection, X_train, X_test, y_train, y_test)

        #     st.write('The f1 score is:', result_dic['f1'])
        #     st.write('The precision score is:', result_dic['precision'])
        #     st.write('The recall score is:', result_dic['recall'])
        #     st.write('The roc auc  is:', result_dic['roc'])


with user_input:

    st.header('Inference using trained model')

    # initializing a form to input user input
    input_form = st.form(key='user')

    user_input_tweet = input_form.text_input(
        'Enter a tweet to check if it is a disaster related tweet or not')
    st.write('**_Your_ input:**', user_input_tweet)

    # when user presses the submit button, it will be stored in this
    user_form_submit_button = input_form.form_submit_button(
        'Submit your input')

    # after user presses the submit button this  then the following steps will execute
    if user_form_submit_button:

        if user_input_tweet == '' or user_input_tweet.isdigit():
            st.write('Input a proper tweet')

        else:

            # cleaning user input text
            cleaned_text = util.user_input_data_cleaning(user_input_tweet)

            if st.session_state['option'] == 'countfid':  # count vectorizer or tfidf
                # count and tfidf

                vectorizer = st.session_state['vectorizer']
                output = vectorizer.transform([cleaned_text])
                user_input = pd.DataFrame(
                    output.todense(), columns=vectorizer.get_feature_names_out())
                st.write('')
                pred = st.session_state['clf_model'].predict(user_input)
                st.write('done')

            # pre trained vectorizer
            if st.session_state['option'] == 'pretrained':


                vectorizer = st.session_state['vectorizer']

                vectorized_input_tweet = util.tweet_vec(
                    cleaned_text, vectorizer)

                vectorized_input_tweet_average = util.average_vec(
                    vectorized_input_tweet)
                vectorized_input_tweet_average = np.reshape(
                    vectorized_input_tweet_average, (1, -1))

                pred = st.session_state['clf_model'].predict(
                    vectorized_input_tweet_average)

            if st.session_state['option'] == 'bert':



                vectorizer = st.session_state['vectorizer']

                vectorized_input_tweet = vectorizer.encode(cleaned_text)

                vectorized_input_tweet = np.reshape(
                    vectorized_input_tweet, (1, -1))

                pred = st.session_state['clf_model'].predict(
                    vectorized_input_tweet)

            if (pred == 0):

                st.markdown('**_Prediction_**: Not a disaster tweet')

            else:
                st.write('**_Prediction_**: Disaster tweet')

 # # setting current directory
        # current_dir=os.path.dirname(__file__)
        # saved_model=os.path.join(current_dir, '.', 'saved_data/saved_model')

        # # loading the saved model from the disk
        # model_load=joblib.load(saved_model)

        # # loading saved vectorizer from disk
        # saved_vectorizer=os.path.join(
        #     current_dir, '.', 'saved_data/saved_vectorizer')
        # vectorizer_load=joblib.load(saved_vectorizer)
