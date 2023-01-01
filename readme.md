# Disaster-tweets-NLP

Twitter disaster classification
Twitter is a platform for news and public discourse amongst people, governments, and corporate entities. There are about 237 million users who regularly share news, opinions, and topics. Additionally, it is a location where people can share incidents and disasters.

Why:
I am a frequent Twitter user who uses it to follow specific people and find a lot of useful information. I chose this dataset from the Twitter First NLP project for that purpose. I chose a basic project to start with, which will help with the basics of NLP.

Issue Statement
The idea of the project was to use the tweets to identify disasters happening in real time.

Use Case
Tweets can be monitored by news or disaster agencies to act quickly.

Project
Data source
The data was picked from Kaggle, one of the leading websites for data science projects and competitions.
Approach- The data is divided into train and test datasets. The total number of tweets was 10,000.

EDA: When I performed explanatory data analysis on the tweets, I found that they contained a lot of stop words, symbols, special characters, and HTML links.  These weren't particularly significant.

Feature Engineering- I used one hot encoding for keyword and location features.

I made an approach by limiting the tweets to main keywords that provide a certain value.

Learning - Utilized Count Vectorizer to convert text into vectors. Count Vectorizer gives a vector to words in sentences and forms.
Which algorithm- RandomForestClassifier
