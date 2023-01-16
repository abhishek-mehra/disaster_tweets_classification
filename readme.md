


# Tweet Classification

### Disaster Tweets Classification

Social media websites have become source of real time news sharing.Twitter and other social media platforms have emerged as popular channels for communication during emergencies.
Twitter is a platform for news and public discourse amongst people, governments, and corporate entities.There are about 237 million users who regularly share news, opinions, and topics on Twitter.

This project aims to classify tweets as either disaster-related or not disaster-related. The data set used includes tweets from the Twitter platform and downloaded from the website( https://appen.com/pre-labeled-datasets/ )
The goal of this project is to build a classifier capable of accurately classifying tweets as disaster-related or not disaster-related.

EDA: When I performed explanatory data analysis on the tweets, I found that the text contained a lot of  symbols, special characters, and HTML links.  These weren't particularly significant. I created helper function to remove the non essential items from the tweets.

The tweets is in text form which cannot be understood by machine learning models, therefore I used Vectorizers to convert the tweets to vectors. I used CountVectorizer, TfIdf ,pretrained Glove  vectors and Sentence transformer based on DistilRoberta to get the vectors.

####  Vectorizers

**CountVectorizers** - It converts a collection of text documents to a matrix of token counts. CountVectorizer creates a matrix in which each unique word is represented by a column of the matrix, and each text sample from the document is a row in the matrix. The value of each cell is nothing but the count of the word in that particular text sample.

**TFIDF vectorizer -** It converts a collection of text documents to a matrix of tf-idf features.Tf-idf stands for Term Frequency Inverse Document Frequency of records. It can be defined as the calculation of how relevant a word in a series or corpus is to a text. The meaning increases proportionally to the number of times in the text a word appears but is compensated by the word frequency in the corpus (data-set).

Pre trained **GLoVe -** GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

GloVe is a technique that takes a large corpus of text data, and uses the co-occurrence statistics of words within that data to train a set of word vectors. These vectors are able to capture the meaning and context of words in a way that can be used as input for machine learning models. It uses the co-occurrence statistics of words to understand their context and relationships and then it represents them in a multi-dimensional vector space.

**SentenceTransformers**  is a Python framework for state-of-the-art sentence, text and image embeddings. I have used [sentence transformer](https://www.notion.so/CO-OP-Applications-933172e1426d4a34b6c18394d82918f4). We used the pretrained **`[distilroberta-base](https://huggingface.co/distilroberta-base)`**
 model and fine-tuned in on a 1B sentence pairs dataset. We use a contrastive learning objective: given a sentence from the pair, the model should predict which out of a set of randomly sampled other sentences, was actually paired with it in our dataset

#### Machine learning models and Metrics

I utilised cross validation to train and evaluate my dataset.I made use of two machine learning models to train my data. RandomForestClassifier and XGBoost were used with varying hyperparameters.

#### Multiple experiments with differnt vectorizers

Using machine learning models on count vectorizer, tfidf vectorizer,glove vectors, I evaluated the models receiving

f1 -score of 0.641 on CountVectorizer and Tfidf vectorizers

f1- score of 0.731 on Glove

f1- score of 0.741 on DistilRoberta

#### Web Application deployment of my model

I deployed my model using a web application. It enables the user to select parameters to construct a machine learning model. It also allows the user to input a tweet to determine if it pertains to a disaster. The web application also provides the user with the ability to view the performance of the model in terms of accuracy, precision, recall and F1 score.
