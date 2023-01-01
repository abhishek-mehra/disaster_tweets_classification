from matplotlib import pyplot as plt
from regex import R
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RANDOM_STATE = 1

def fill_na(df):
    df['keyword'].fillna(df['keyword'].mode()[0],inplace=True)   #replacing NaN in keyword with mode values


    df['location'].fillna(df['location'].mode()[0],inplace=True)  #replacing NaN in location with mode values

    return df


def trim_spaces(df):
    df['location'] = df['location'].str.strip()

    return df


def make_dummies(df):
    gd = pd.get_dummies(df['keyword'], prefix = ['keyword'])


    gd2 = pd.get_dummies(df['location'], prefix = ['location'])

    return gd, gd2




def clean_vectorize_using_count_vectorizer(df, text_col):
    """Convert text column to columns of numbers.

    df[text_col] =>>>>>> df [["feat1", "feat2"......]]

    Args:
        df (_type_): _description_
        text_col (_type_): _description_

    Returns:
        _type_: _description_
    """
    cv = CountVectorizer()  #creating cv ,CountVectorizer object
    cv_train = cv.fit_transform(df[text_col])
    df = pd.DataFrame(cv_train.todense(),columns = cv.get_feature_names_out())

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
    df = pd.DataFrame(tf_train.todense(), columns = tf.get_feature_names_out())
    return df


def remove_url(text):
    url=re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)   #removing html texts

def remove_punct(text):
    table = str.maketrans('','',string.punctuation)
    return text.translate(table)

def remove_emoji(text):
    emoji_pattern = re.compile('['
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                ']+',flags=re.UNICODE)
    return emoji_pattern.sub(r'',text)

def clean_tweets(df, text_col):
    df[text_col]=df[text_col].apply(lambda x:remove_html(x))   #apply lambda to all values in df series
    df[text_col]=df[text_col].apply(lambda x:remove_url(x))  #pass a function and apply it to every single value of the series
    df[text_col]=df[text_col].apply(lambda x:remove_emoji(x))
    df[text_col]=df[text_col].apply(lambda x: remove_punct(x))
    return df

def location_occurence(df):
    top_10 = df['location'].value_counts()[:10]
    fig = plt.figure(figsize = (10,4))
    top_10.plot.barh()
    return fig



def keyword_occurence(df):
    top_10 = df['keyword'].value_counts()[:10]
    fig = plt.figure(figsize = (10,4))
    top_10.plot.barh()
    return fig


def words_distribution(df):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

    disaster_tweets = df[df['target']==1]['text']
    tweets = disaster_tweets.str.split()
    number_of_words = tweets.map(lambda x:len(x))
    ax1.hist(number_of_words,color='red')
    ax1.set_title('Disaster tweets')
    ax1.set_xlabel('Number of words')
    ax1.set_ylabel('Number of tweets')

    non_disaster_tweets = df[df['target']==0]['text']
    tweets = non_disaster_tweets.str.split()
    number_of_words = tweets.map(lambda x: len(x))
    ax2.hist(number_of_words,color='green')
    ax2.set_title('Not disaster tweets')
    fig.suptitle('Words in a tweet')



def ttsplit(df, label_col_name='target', test_size=0.2):
    y = df[label_col_name]
    df = df.drop(label_col_name, axis=1, inplace=False)
    assert label_col_name not in df.columns

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=test_size, shuffle=True, random_state=RANDOM_STATE, stratify=y)

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
