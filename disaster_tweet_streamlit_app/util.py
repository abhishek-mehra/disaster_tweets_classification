from matplotlib import pyplot as plt
from regex import R
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from xgboost import train
import streamlit as st
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords

RANDOM_STATE = 1



def clean_vectorize_using_count_vectorizer(df, text_col):
    """Convert text column to columns of numbers.

    df[text_col] =>>>>>> df [["feat1", "feat2"......]]

    Args:
        df (_type_): _description_
        text_col (_type_): _description_

    Returns:
        _type_: _description_
    """
    cv = CountVectorizer()  # creating cv ,CountVectorizer object
    cv_train = cv.fit_transform(df[text_col])
    df = pd.DataFrame(cv_train.todense(), columns=cv.get_feature_names_out())

    return df



def clean_vectorize_using_tfidf_vectorizer(df, text_col):
    """Convert text column to columns of numbers.

    df[text_col] =>>>>>> df [["feat1", "feat2"......]]

    Args:
        df (_type_): _description_
        text_col (_type_): _description_

    Returns:
        _type_: _description_
    """
    tf = TfidfVectorizer(tokenizer=word_tokenize)
    tf_train = tf.fit_transform(df[text_col])
    df = pd.DataFrame(tf_train.todense(), columns=tf.get_feature_names_out())
    return df


# HTML removal
def remove_html(text):
    pattern = re.compile(r'<.*?>')
    return pattern.sub(r'', text)


# URL removal

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)


# Lowercasing

def lowercasing(text):
    return text.str.lower()


# Punctuation removal

def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)
# Stopwords removal


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    d = ''
    for i in text.split():
        if i not in stop_words:
            d = d + ' ' + i
    return d.strip()


# Replace number with a tag

def replace_number_with_tag(text):
    tag = '#'
    a = ''
    for i in text.split():
        # print (i)
        if i.isdigit():
            a = a + ' ' + tag
        else:
            a = a + ' ' + i

    return a.strip()


# Lemmatization

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    combined = ''
    for word in text.split():
        # print (word)
        combined = combined + ' ' + lemmatizer.lemmatize(word)
        # print (combined)

    return combined.strip()


# Stemming
def stemming_porter(text):
    combined = ''
    stemmer = PorterStemmer()
    for word in text.split():
        combined = combined + ' ' + stemmer.stem(word)

    return combined.strip()

 # Separating number and words together


def separate_num_word(text):
    new_text = ''
    for i in text.split():
        w = ''
        n = ''

        if not i.isalpha():

            for c in i:
                if c.isdigit():
                    n = n+c
                else:
                    w = w+c

            new_text = new_text + ' ' + n + ' ' + w

        else:
            new_text = new_text + ' ' + i

    return new_text.strip()



def data_cleaning(df):
    df2 = df.copy(deep=True)
    df2['cleaned_text'] = df2['text'].apply(lambda x: remove_html(x))
    df2['cleaned_text'] = df2['cleaned_text'].apply(lambda x: remove_url(x))
    df2['cleaned_text'] = lowercasing(df2['cleaned_text'])
    df2['cleaned_text'] = df2['cleaned_text'].apply(lambda x: remove_punct(x))
    df2['cleaned_text'] = df2['cleaned_text'].apply(
        lambda x: remove_stopwords(x))
    df2['cleaned_text'] = df2['cleaned_text'].apply(
        lambda x: separate_num_word(x))
    df2['cleaned_text'] = df2['cleaned_text'].apply(
        lambda x: replace_number_with_tag(x))
    # df2['cleaned_text'] = df2['cleaned_text'].apply(lambda x:remove_nonenglish(x))
    df2['cleaned_text'] = df2['cleaned_text'].apply(lambda x: lemmatization(x))
    # df2['cleaned_text'] = df2['cleaned_text'].apply(lambda x:stemming_porter(x))

    return df2


def location_occurence(df):
    top_10 = df['location'].value_counts()[:10]
    fig = plt.figure(figsize=(10, 4))
    top_10.plot.barh()
    return fig


def keyword_occurence(df):
    top_10 = df['keyword'].value_counts()[:10]
    fig = plt.figure(figsize=(10, 4))
    top_10.plot.barh()
    return fig


def words_distribution(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    disaster_tweets = df[df['target'] == 1]['text']
    tweets = disaster_tweets.str.split()
    number_of_words = tweets.map(lambda x: len(x))
    ax1.hist(number_of_words, color='red')
    ax1.set_title('Disaster tweets')
    ax1.set_xlabel('Number of words')
    ax1.set_ylabel('Number of tweets')

    non_disaster_tweets = df[df['target'] == 0]['text']
    tweets = non_disaster_tweets.str.split()
    number_of_words = tweets.map(lambda x: len(x))
    ax2.hist(number_of_words, color='green')
    ax2.set_title('Not disaster tweets')
    fig.suptitle('Words in a tweet')



def ttsplit(df, label_col_name='target', feature_column=None, test_size=0.2):

    y = df[label_col_name]
    df = df.drop(label_col_name, axis=1, inplace=False)
    assert label_col_name not in df.columns

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=test_size, shuffle=True, random_state=RANDOM_STATE, stratify=y)

    if feature_column:
        X_train = X_train[feature_column]
        X_test = X_test[feature_column]

    return X_train, X_test, y_train, y_test



def training_eval(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # dataframe containing just y_test and predictions
    output_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

    # miss classified data - false positives
    fp_df = X_test.loc[(output_df['Actual'] == 0) &
                       (output_df['Predicted'] == 1)]

    # miss classified data - false negatives
    fn_df = X_test.loc[(output_df['Actual'] == 1) &
                       (output_df['Predicted'] == 0)]

    output_dic = {'f1': round(f1_score(y_test, predictions), 3),
                  'precision': round(precision_score(y_test, predictions), 3),
                  'recall': round(recall_score(y_test, predictions), 3),
                  'classification': classification_report(y_test, predictions, output_dict=True),
                  'false_positves': fp_df,
                  'false_negatives': fn_df}

    return output_dic
