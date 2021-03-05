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


# p2.6 d
stop_words = set(stopwords.words('english')) 
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
		trainTweet.append(row[1].lower())


for i in range(len(trainTweet)):
	unprocess = word_tokenize(trainTweet[i])
	trainTweet[i] = [w for w in unprocess if not w in stop_words]


# wvModel = Word2Vec(trainTweet, size=300)
# wvModel.save("wvmodel.model")
average = []
wvModel = Word2Vec.load("wvmodel.model")
# print(wvModel["guys"])
for tweet in range(len(trainTweet)):
	count = 0
	vec = []
	for word in trainTweet[tweet]:
		if word in wvModel.wv.vocab:
			count += 1
			vec.append(wvModel[word])
	if count != 0:
		average.append(sum(vec)/count)
	else:
		trainSentiment.pop(tweet)
# pprint(average, indent=4)
clf = LogisticRegression(max_iter=60000)
clf.fit(average, trainSentiment)

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
		testTweet.append(row[1].lower())

for i in range(len(testTweet)):
	unprocess = word_tokenize(testTweet[i])
	testTweet[i] = [w for w in unprocess if not w in stop_words]

testVec = Word2Vec(testTweet, size=300)
testTweetVec = []

for tweet in range((len(testTweet))):
	count = 0
	vec = []
	for word in testTweet[tweet]:
		if word in testVec.wv.vocab:
			count += 1
			vec.append(testVec[word])
	if count != 0:
		testTweetVec.append(sum(vec)/count)
	else:
		testSentiment.pop(tweet)
testPredict = []
for tweet in testTweetVec:
	testPredict.append(clf.predict(tweet.reshape(1, -1)))
print(testPredict[0][0])
count = 0
for i in range(len(testPredict)):
	if str(testPredict[i][0]) == str(testSentiment[i]):
		count += 1
print(count/len(testPredict))


# # p2.6 a b c
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
# 		trainTweet.append(row[1].lower())


# for i in range(len(trainTweet)):
# 	trainTweet[i] = word_tokenize(trainTweet[i])


# # wvModel = Word2Vec(trainTweet, size=300)
# # wvModel.save("wvmodel.model")
# average = []
# wvModel = Word2Vec.load("wvmodel.model")
# # print(wvModel["guys"])
# for tweet in range(len(trainTweet)):
# 	count = 0
# 	vec = []
# 	for word in trainTweet[tweet]:
# 		if word in wvModel.wv.vocab:
# 			count += 1
# 			vec.append(wvModel[word])
# 	if count != 0:
# 		average.append(sum(vec)/count)
# 	else:
# 		trainSentiment.pop(tweet)
# # pprint(average, indent=4)
# clf = LogisticRegression(max_iter=60000)
# clf.fit(average, trainSentiment)

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
# 		testTweet.append(row[1].lower())

# for i in range(len(testTweet)):
# 	testTweet[i] = word_tokenize(testTweet[i])

# testVec = Word2Vec(testTweet, size=300)
# testTweetVec = []

# for tweet in range((len(testTweet))):
# 	count = 0
# 	vec = []
# 	for word in testTweet[tweet]:
# 		if word in testVec.wv.vocab:
# 			count += 1
# 			vec.append(testVec[word])
# 	if count != 0:
# 		testTweetVec.append(sum(vec)/count)
# 	else:
# 		testSentiment.pop(tweet)
# testPredict = []
# for tweet in testTweetVec:
# 	testPredict.append(clf.predict(tweet.reshape(1, -1)))
# print(testPredict[0][0])
# count = 0
# for i in range(len(testPredict)):
# 	if str(testPredict[i][0]) == str(testSentiment[i]):
# 		count += 1
# print(count/len(testPredict))
# # 0.49303621169916434




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




