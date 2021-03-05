import csv
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from pprint import pprint
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) 
trainSentiment = []
trainTweet = []
vectorizer = CountVectorizer(stop_words="english", max_features=4000)
count = 0
with open("trainingandtestdata/training.1600000.processed.noemoticon.csv", "r", encoding='latin-1') as f:
	reader = csv.reader(f)
	for row in reader:
		count += 1
		if count == 1:
			continue
		trainSentiment.append(row[0])
		tokens = word_tokenize(row[5].lower())
		processedToken = [w for w in tokens if not w in stop_words]
		trainTweet.append(processedToken)
trainTweet = vectorizer.fit_transform(trainTweet)
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
testTweet = vectorizer.transform(testTweet)
# print(str(clf.predict(testTweet[0])[0]) == str(testSentiment[200]))
count = 0
for i in range(len(testSentiment)):
	if str(clf.predict(testTweet[i])[0]) == str(testSentiment[i]):
		count += 1
print(count/len(testSentiment))
