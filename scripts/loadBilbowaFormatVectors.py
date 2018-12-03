#!python3
import numpy as np

def load(inputFile):
	'''
	Returns {string:tuple}.
	'''
	table = {}
	with open(inputFile, 'r', encoding='utf-8') as fin:
		line = fin.readline()
		num, dim = line.split()
		num = int(num)
		dim = int(dim)
		for i, line in enumerate(fin):
			parts = line.split()
			word = parts[0]
			vector = parts[1:]
			assert(len(vector) == dim)
			vector = tuple(float(e) for e in vector)
			table[word] = vector
		if i+1 != num:
			print('Warning: claiming {0} word vectors but found {1}.'.format(num, i+1))
			num = i+1
		assert(len(table) == num)
	return table

def normalizeColumn(a):
	s = a**2
	s = s.sum(axis=0)
	s = np.sqrt(s)
	s[s == 0] = 1
	return a/s

def loadAndNormalize(inputFile):
	'''
	Returns {string:numpy array}, with normalized vectors.
	'''
	table = {}
	words = []
	vectors = []
	with open(inputFile, 'r', encoding='utf-8') as fin:
		line = fin.readline()
		num, dim = line.split()
		num = int(num)
		dim = int(dim)
		for i, line in enumerate(fin):
			parts = line.split()
			word = parts[0]
			vector = parts[1:]
			assert(len(vector) == dim)
			vector = tuple(float(e) for e in vector)
			words.append(word)
			vectors.append(vector)
		if i+1 != num:
			print('Warning: claiming {0} word vectors but found {1}.'.format(num, i+1))
			num = i+1
		matrix = np.array(vectors)
		matrix = normalizeColumn(matrix.T)
		for i in range(num):
			table[words[i]] = matrix[:, i]
		assert(len(table) == num)
	return table
