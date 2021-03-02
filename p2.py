import csv
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from pprint import pprint
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer

trainSentiment = []
trainTweet = []
vectorizer = CountVectorizer(stop_words="english", max_features=1000)
with open("sentiment-train.csv", "r") as f:
	reader = csv.reader(f)
	for row in reader:
		trainSentiment.append(row[0])
		trainTweet.append(row[1])
trainTweet = vectorizer.fit_transform(trainTweet)
# trainSentiment = vectorizer.fit_transform(trainSentiment)
# print(trainTweet)
clf = MultinomialNB()
clf.fit(trainTweet, trainSentiment)

testSentiment = []
testTweet = []
with open()




