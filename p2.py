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
import spacy
from gensim.models import Word2Vec

# p2.6
trainSentiment = []
trainTweet = ""
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
count = 0
with open("sentiment-train.csv", "r") as f:
	reader = csv.reader(f)
	for row in reader:
		count += 1
		if count == 1:
			continue
		trainSentiment.append(row[0])
		trainTweet = trainTweet + " " + row[1]
trainTweet = trainTweet.lower()
lang_class = spacy.util.get_lang_class('en')
nlp = lang_class()
nlp.max_length = 5425470
tokens = nlp.make_doc(trainTweet)
# pprint(tokens[:10], indent=4)
wvModel = Word2Vec(trainTweet, size=300)

# pprint(wvModel.train(trainTweet), indent=4)
wvModel.save("wvmodel.model")



# # p2.5:4000, 0.766016713091922
# trainSentiment = []
# trainTweet = []
# vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
# count = 0
# with open("sentiment-train.csv", "r") as f:
# 	reader = csv.reader(f)
# 	for row in reader:
# 		count += 1
# 		if count == 1:
# 			continue
# 		trainSentiment.append(row[0])
# 		trainTweet.append(row[1])
# maxFreature = [1000, 2000, 3000, 4000]
# res = []
# for feature in maxFreature:
# 	vectorizer = TfidfVectorizer(stop_words="english", max_features=feature)
# 	trainTweetDone = vectorizer.fit_transform(trainTweet)
# 	clf = MultinomialNB()
# 	res.append(sum(cross_val_score(clf, trainTweetDone, trainSentiment, cv=5, scoring="accuracy"))/5)
# print(res)
# print("the max feature is", maxFreature[maxFreature.index(max(maxFreature))])
# # a done
# vectorizer = TfidfVectorizer(stop_words="english", max_features=maxFreature[maxFreature.index(max(maxFreature))])
# trainTweet = vectorizer.fit_transform(trainTweet)
# clf = MultinomialNB()
# clf.fit(trainTweet, trainSentiment)

# testSentiment = []
# testTweet = []
# count = 0
# with open("sentiment-test.csv", "r") as f:
# 	reader = csv.reader(f)
# 	for row in reader:
# 		count += 1
# 		if count == 1:
# 			continue
# 		testSentiment.append(row[0])
# 		testTweet.append(row[1])
# testTweet = vectorizer.transform(testTweet)

# count = 0
# for i in range(len(testSentiment)):
# 	# print(str(clf.predict(testTweet[i])))
# 	# print(str(testSentiment[i]))
# 	if str(clf.predict(testTweet[i])[0]) == str(testSentiment[i]):
# 		count += 1
# print(count/len(testSentiment))






# # p2.4:0.7688022284122563
# trainSentiment = []
# trainTweet = []
# vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
# count = 0
# with open("sentiment-train.csv", "r") as f:
# 	reader = csv.reader(f)
# 	for row in reader:
# 		count += 1
# 		if count == 1:
# 			continue
# 		trainSentiment.append(row[0])
# 		trainTweet.append(row[1])
# trainTweet = vectorizer.fit_transform(trainTweet)
# # trainSentiment = vectorizer.fit_transform(trainSentiment)
# # print(trainTweet)
# clf = LogisticRegression()
# clf.fit(trainTweet, trainSentiment)

# testSentiment = []
# testTweet = []
# count = 0
# with open("sentiment-test.csv", "r") as f:
# 	reader = csv.reader(f)
# 	for row in reader:
# 		count += 1
# 		if count == 1:
# 			continue
# 		testSentiment.append(row[0])
# 		testTweet.append(row[1])
# testTweet = vectorizer.transform(testTweet)
# # print(str(clf.predict(testTweet[0])[0]) == str(testSentiment[200]))
# count = 0
# for i in range(len(testSentiment)):
# 	if str(clf.predict(testTweet[i])[0]) == str(testSentiment[i]):
# 		count += 1
# print(count/len(testSentiment))



# # p2.3: 0.766016713091922
# trainSentiment = []
# trainTweet = []
# vectorizer = CountVectorizer(stop_words="english", max_features=1000)
# count = 0
# with open("sentiment-train.csv", "r") as f:
# 	reader = csv.reader(f)
# 	for row in reader:
# 		count += 1
# 		if count == 1:
# 			continue
# 		trainSentiment.append(row[0])
# 		trainTweet.append(row[1])
# trainTweet = vectorizer.fit_transform(trainTweet)
# # trainSentiment = vectorizer.fit_transform(trainSentiment)
# # print(trainTweet)
# clf = LogisticRegression()
# clf.fit(trainTweet, trainSentiment)

# testSentiment = []
# testTweet = []
# count = 0
# with open("sentiment-test.csv", "r") as f:
# 	reader = csv.reader(f)
# 	for row in reader:
# 		count += 1
# 		if count == 1:
# 			continue
# 		testSentiment.append(row[0])
# 		testTweet.append(row[1])
# testTweet = vectorizer.transform(testTweet)
# # print(str(clf.predict(testTweet[0])[0]) == str(testSentiment[200]))
# count = 0
# for i in range(len(testSentiment)):
# 	if str(clf.predict(testTweet[i])[0]) == str(testSentiment[i]):
# 		count += 1
# print(count/len(testSentiment))



# # p2.2:0.7688022284122563
# trainSentiment = []
# trainTweet = []
# vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
# count = 0
# with open("sentiment-train.csv", "r") as f:
# 	reader = csv.reader(f)
# 	for row in reader:
# 		count += 1
# 		if count == 1:
# 			continue
# 		trainSentiment.append(row[0])
# 		trainTweet.append(row[1])
# trainTweet = vectorizer.fit_transform(trainTweet)
# # trainSentiment = vectorizer.fit_transform(trainSentiment)
# # print(trainTweet)
# clf = MultinomialNB()
# clf.fit(trainTweet, trainSentiment)

# testSentiment = []
# testTweet = []
# count = 0
# with open("sentiment-test.csv", "r") as f:
# 	reader = csv.reader(f)
# 	for row in reader:
# 		count += 1
# 		if count == 1:
# 			continue
# 		testSentiment.append(row[0])
# 		testTweet.append(row[1])
# testTweet = vectorizer.transform(testTweet)
# # print(str(clf.predict(testTweet[0])[0]) == str(testSentiment[200]))
# count = 0
# for i in range(len(testSentiment)):
# 	if str(clf.predict(testTweet[i])[0]) == str(testSentiment[i]):
# 		count += 1
# print(count/len(testSentiment))



# # p2.1: 0.7827298050139275
# trainSentiment = []
# trainTweet = []
# vectorizer = CountVectorizer(stop_words="english", max_features=1000)
# count = 0
# with open("sentiment-train.csv", "r") as f:
# 	reader = csv.reader(f)
# 	for row in reader:
# 		count += 1
# 		if count == 1:
# 			continue
# 		trainSentiment.append(row[0])
# 		trainTweet.append(row[1])
# trainTweet = vectorizer.fit_transform(trainTweet)
# # trainSentiment = vectorizer.fit_transform(trainSentiment)
# # print(trainTweet)
# clf = MultinomialNB()
# clf.fit(trainTweet, trainSentiment)

# testSentiment = []
# testTweet = []
# count = 0
# with open("sentiment-test.csv", "r") as f:
# 	reader = csv.reader(f)
# 	for row in reader:
# 		count += 1
# 		if count == 1:
# 			continue
# 		testSentiment.append(row[0])
# 		testTweet.append(row[1])
# testTweet = vectorizer.transform(testTweet)
# # print(str(clf.predict(testTweet[0])[0]) == str(testSentiment[200]))
# count = 0
# for i in range(len(testSentiment)):
# 	if str(clf.predict(testTweet[i])[0]) == str(testSentiment[i]):
# 		count += 1
# print(count/len(testSentiment))




