#!python3

import sys
import operator
import math
import loadBilbowaFormatVectors
import numpy as np

def dotProduct(v1, v2):
	return sum(map(operator.mul, v1, v2))

def cosSim(v1, v2, sqnorm1, sqnorm2):
	return dotProduct(v1, v2) / math.sqrt(sqnorm1 * sqnorm2)

def knn(srcTable, tgtTable, word, K, sim):
	'''
	Returns a list (maximum length @K) of (target word, similarity) tuples if @word is in @srcTable,
	or None otherwise.
	'''
	if word in srcTable:
		srcVec = srcTable[word]
		sqnormSrcVec = dotProduct(srcVec, srcVec)
		scores = []
		for k, v in tgtTable.items():
			score = sim(srcVec, v, sqnormSrcVec, dotProduct(v, v))
			scores.append((k, score))
		sortedScores = sorted(scores, key=lambda kv: kv[1], reverse=True)
		return sortedScores[0:K]
	return None

def knnWithNormalizedVectors(srcTable, tgtTable, word, K):
	'''
	This function assumes the vectors in the tables are normalized numpy arrays, so cosine similarity becomes dot product.
	'''
	if word in srcTable:
		srcVec = srcTable[word]
		scores = []
		for k, v in tgtTable.items():
			score = np.dot(srcVec, v)
			scores.append((k, score))
		sortedScores = sorted(scores, key=lambda kv: kv[1], reverse=True)
		return sortedScores[0:K]
	return None

if __name__ == '__main__':
	srcTable = loadBilbowaFormatVectors.load(sys.argv[1])
	tgtTable = loadBilbowaFormatVectors.load(sys.argv[2])
	K = int(sys.argv[3])
	for line in sys.stdin:
		KList = knn(srcTable, tgtTable, line.strip(), K, cosSim)
		if KList is None:
			print('Unknown word.')
		else:
			for k, v in KList:
				print(k, v)