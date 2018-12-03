#!python3

import sys
import loadBilbowaFormatVectors
import knn

if __name__ == '__main__':
	srcTable = loadBilbowaFormatVectors.loadAndNormalize(sys.argv[1])
	tgtTable = loadBilbowaFormatVectors.loadAndNormalize(sys.argv[2])
	if '</s>' in tgtTable:
		tgtTable.pop('</s>')
	srcWordFile = sys.argv[3]
	outputFile = sys.argv[4]
	K = 10 #len(tgtTable)
	with open(srcWordFile, 'r', encoding='utf-8') as fin, \
		open(outputFile, 'w', encoding='utf-8') as fout:
		for line in fin:
			srcWord = line.strip()
			KList = knn.knnWithNormalizedVectors(srcTable, tgtTable, srcWord, K)
			if KList is None:
				print('Unknown word:', srcWord)
			else:
				fout.write('{0}\t{1}\n'.format(srcWord, ' '.join(k for k, _ in KList)))
