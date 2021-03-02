import csv
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from pprint import pprint
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer

trainSentiment = []
trainTweet = []
vectorizer = CountVectorizer(stop_words="english", max_features=1000)
count = 0
with open("sentiment-train.csv", "r") as f:
	reader = csv.reader(f)
	for row in reader:
		count += 1
		if count == 1:
			continue
		trainSentiment.append(row[0])
		trainTweet.append(row[1])
trainTweet = vectorizer.fit_transform(trainTweet)
# trainSentiment = vectorizer.fit_transform(trainSentiment)
# print(trainTweet)
clf = MultinomialNB()
clf.fit(trainTweet, trainSentiment)

testSentiment = []
testTweet = []
count = 0
with open("sentiment-test.csv", "r") as f:
	reader = csv.reader(f)
	for row in reader:
		count += 1
		if count == 1:
			continue
		testSentiment.append(row[0])
		testTweet.append(row[1])
testTweet = vectorizer.fit_transform(testTweet)
# print(str(clf.predict(testTweet[0])[0]) == str(testSentiment[200]))
count = 0
for i in range(len(testSentiment)):
	if str(clf.predict(testTweet[i])[0]) == str(testSentiment[0]):
		count += 1
print(count/len(testSentiment))




