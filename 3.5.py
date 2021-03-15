import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from pprint import pprint
from sklearn.decomposition import PCA
import csv
from matplotlib import pyplot
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# 3.5
with open("vocab.txt", "r") as f:
	vocab = f.readlines()
for i in range(len(vocab)):
	vocab[i] = vocab[i].rstrip()

wwMatrix = np.zeros((len(vocab), len(vocab)))

sentences = []
with open("will_play_text.csv", "r") as f:
	reader = csv.reader(f, delimiter=";")
	for row in reader:
		sentence = nltk.sent_tokenize(row[5])
		for s in sentence:
			sentences.append(s)
words = []
for sent in sentences:
	words.append(nltk.word_tokenize(sent))
# print(words)
for sent in words:
	windex = []
	for w in sent:
		if w in vocab:
			windex.append(vocab.index(w))
	for index in windex:
		addIndex = windex
		addIndex.remove(index)
		for i in addIndex:
			wwMatrix[index][i] += 1
with open("wwMatrix.csv", "w") as f:
	csvw = csv.writer(f)
	csvw.writerows(wwMatrix)


# # 3.6
# wwMatrix = []
# with open("wwMatrix.csv", "r") as f:
# 	reader = csv.reader(f)
# 	for row in reader:
# 		wwMatrix.append(row)
# print(len(wwMatrix))
# playName = open("play_names.txt", "r")
# plays_comedy = ["The Tempest", "Two Gentlemen of Verona", "Merry Wives of Windsor", "Measure for measure", "A Comedy of Errors", "Much Ado about nothing", "Loves Labours Lost", "A Midsummer nights dream", "Merchant of Venice", "As you like it", "Taming of the Shrew", "Alls well that ends well", "Twelfth Night", "A Winters Tale", "Pericles"]
# plays_history = ["King John", "Richard II", "Richard III", "Henry IV", "Henry VI Part 2", "Henry VIII", "Henry VI Part 1", "Henry V", "Henry VI Part 3"]
# plays_tragedy = ["Troilus and Cressida", "Coriolanus", "Titus Andronicus", "Romeo and Juliet", "Timon of Athens", "Julius Caesar", "macbeth", "Hamlet", "King Lear", "Othello", "Antony and Cleopatra", "Cymbeline"]
# play_doc = {}
# doc = pd.read_csv("will_play_text.csv", sep=";", header=None)
# for play in plays_comedy:
# 	lines = []
# 	for row in range(len(doc)):
# 		if doc[1][row] == play:
# 			lines.append(doc[5][row])
# 	play_doc[play] = lines

# play_token = {}
# for play in plays_comedy:
# 	play_token[play] = []
# 	for sentence in play_doc[play]:
# 		play_token[play] = play_token[play] + nltk.word_tokenize(sentence)
# # print(play_token["A Midsummer nights dream"])
# with open("vocab.txt", "r") as f:
# 	vocab = f.readlines()
# for i in range(len(vocab)):
# 	vocab[i] = vocab[i].rstrip()

# play_comedy_vec = {}
# for play in plays_comedy:
# 	vec = np.zeros(22602)
# 	# print(vec)
# 	count = 0
# 	for w in play_token[play]:
# 		if w in vocab:
# 			count += 1
# 			wordvec = np.array(wwMatrix[vocab.index(w)]).astype(float)
# 			vec = vec + wordvec
# 			# print(vec)
# 	play_comedy_vec[play] = vec/count
# print(play_comedy_vec["A Midsummer nights dream"])











# # 3.3, 4
# doc = pd.read_csv("will_play_text.csv", sep=";", header=None)
# playName = open("play_names.txt", "r")
# plays = []
# for name in playName:
# 	plays.append(name.rstrip())

# line = [""] * len(plays)


# for row in range(len(doc)):
# 	line[plays.index(doc[1][row])] = line[plays.index(doc[1][row])] + " " + doc[5][row]

# vectorizer = TfidfVectorizer()
# m = vectorizer.fit_transform(line)
# # df = pd.DataFrame(m.toarray(), columns=vectorizer.get_feature_names(), index=plays)
# df = pd.DataFrame(m.toarray())
# df = df
# df.to_csv("term_doc_matrix.csv")
# print(df)

# tdMatrix = []
# with open('term_doc_matrix.csv', 'r') as file:
# 	reader = csv.reader(file)
# 	count = 0 
# 	for row in reader:
# 		count += 1
# 		if count == 1:
# 			continue
# 		row.pop(0)
# 		tdMatrix.append(row)
# pca = PCA(n_components=2)
# # print(tdMatrix)
# res = pca.fit_transform(tdMatrix)
# pyplot.scatter(res[:, 0], res[:, 1])
# for i in range(len(plays)):
# 	pyplot.annotate(plays[i], xy=(res[i, 0], res[i, 1]))
# pyplot.show()



# # 3.1,2
# doc = pd.read_csv("will_play_text.csv", sep=";", header=None)
# playName = open("play_names.txt", "r")
# plays = []
# for name in playName:
# 	plays.append(name.rstrip())

# line = [""] * len(plays)


# for row in range(len(doc)):
# 	line[plays.index(doc[1][row])] = line[plays.index(doc[1][row])] + " " + doc[5][row]

# vectorizer = CountVectorizer()
# m = vectorizer.fit_transform(line)
# # df = pd.DataFrame(m.toarray(), columns=vectorizer.get_feature_names(), index=plays)
# df = pd.DataFrame(m.toarray())
# df = df
# df.to_csv("term_doc_matrix.csv")
# # print(df)

# tdMatrix = []
# with open('term_doc_matrix.csv', 'r') as file:
# 	reader = csv.reader(file)
# 	count = 0 
# 	for row in reader:
# 		count += 1
# 		if count == 1:
# 			continue
# 		row.pop(0)
# 		tdMatrix.append(row)

# pca = PCA(n_components=2)
# # print(tdMatrix)
# res = pca.fit_transform(tdMatrix)
# # print(len(res))
# pyplot.scatter(res[:, 0], res[:, 1])
# for i in range(len(plays)):
# 	pyplot.annotate(plays[i], xy=(res[i, 0], res[i, 1]))
# pyplot.show()