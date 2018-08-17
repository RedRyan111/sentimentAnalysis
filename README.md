#sentiment analysis
Used following repositories for scraping date:
twitter data: https://github.com/Jefferson-Henrique/GetOldTweets-python
reddit data: https://github.com/pushshift/api
The Bash script runs the twitterApi.py and redditApi.py files in there respective folders.
These files collect tweets and reddit posts and put them into a txt and csv file.
The model folder holds the model python files that will train on the sentiment140 dataset
of 1.6 million tweets. Some parameters need to be changed and tested to get full word embeddings
and full dataset. Comments are put in place to help with future edditing. The model is a manually built
LSTM network done in Tensorflow. Downloads required, including GloveWord embeddings, sentiment140 dataset
and many python dependencies.
