import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from pprint import pprint
from sklearn.decomposition import PCA
import csv
from matplotlib import pyplot
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# 3.3, 4
doc = pd.read_csv("will_play_text.csv", sep=";", header=None)
playName = open("play_names.txt", "r")
plays = []
for name in playName:
	plays.append(name.rstrip())

line = [""] * len(plays)


for row in range(len(doc)):
	line[plays.index(doc[1][row])] = line[plays.index(doc[1][row])] + " " + doc[5][row]

vectorizer = TfidfVectorizer()
m = vectorizer.fit_transform(line)
# df = pd.DataFrame(m.toarray(), columns=vectorizer.get_feature_names(), index=plays)
df = pd.DataFrame(m.toarray())
df = df
df.to_csv("term_doc_matrix.csv")
print(df)

tdMatrix = []
with open('term_doc_matrix.csv', 'r') as file:
	reader = csv.reader(file)
	count = 0 
	for row in reader:
		count += 1
		if count == 1:
			continue
		row.pop(0)
		tdMatrix.append(row)
pca = PCA(n_components=2)
# print(tdMatrix)
res = pca.fit_transform(tdMatrix)
pyplot.scatter(res[:, 0], res[:, 1])
for i in range(len(plays)):
	pyplot.annotate(plays[i], xy=(res[i, 0], res[i, 1]))
pyplot.savefig("3.4.png")