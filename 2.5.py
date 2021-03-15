import csv
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from pprint import pprint
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from nltk import word_tokenize
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords

# p2.5:4000, 0.766016713091922
trainSentiment = []
trainTweet = []
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
count = 0
with open("sentiment-train.csv", "r") as f:
	reader = csv.reader(f)
	for row in reader:
		count += 1
		if count == 1:
			continue
		trainSentiment.append(row[0])
		trainTweet.append(row[1])
maxFreature = [1000, 2000, 3000, 4000]
res = []
for feature in maxFreature:
	vectorizer = TfidfVectorizer(stop_words="english", max_features=feature)
	trainTweetDone = vectorizer.fit_transform(trainTweet)
	clf = MultinomialNB()
	res.append(sum(cross_val_score(clf, trainTweetDone, trainSentiment, cv=5, scoring="accuracy"))/5)
print(res)
print("the max feature is", maxFreature[maxFreature.index(max(maxFreature))])
# a done
vectorizer = TfidfVectorizer(stop_words="english", max_features=maxFreature[maxFreature.index(max(maxFreature))])
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

count = 0
for i in range(len(testSentiment)):
	# print(str(clf.predict(testTweet[i])))
	# print(str(testSentiment[i]))
	if str(clf.predict(testTweet[i])[0]) == str(testSentiment[i]):
		count += 1
print(count/len(testSentiment))