# UBiLexAT: An Unsupervised Bilingual Lexicon Inducer From Non-Parallel Data by Adversarial Training #

This software can produce a bilingual lexicon from non-parallel data without any cross-lingual supervision. It does so by learning a transformation between source word embeddings and target ones by adversarial training. The technique is described in the following paper:

> Meng Zhang, Yang Liu, Huanbo Luan, and Maosong Sun. Adversarial Training for Unsupervised Bilingual Lexicon Induction. In Proceedings of ACL, 2017.

## Runtime Environment ##

This software has been tested in the following environment, but should work in a compatible one.

- 64-bit Linux
- Python 2.7 (for adversarial training code in the `src` folder)
	- Theano
	- Lasagne ([bleeding-edge version](http://lasagne.readthedocs.io/en/latest/user/installation.html#bleeding-edge-version) needed as of April, 2017)
	- scikit-learn
- Python 3.4 (for translation code in the `scripts` folder)

## Usage ##

1\. Specify the variables in the `config` file. For example, if `config` contains the following lines:

	config=zh-en
	lang1=zh
	lang2=en

then the data should be located in `data/zh-en` with file extensions `zh` and `en`.

2\. Prepare the following data in the folder specified in Step 1:

- word2vec.zh/en: Word embeddings, which can be obtained by running word2vec on monolingual data.
- vocab-freq.zh/en: Space-separated word-frequency pairs.

We provide the data for the five language pairs considered in our paper.

3\. Train and obtain the bilingual lexicon.

`./run.sh`

4\. The following files will be generated in `data/zh-en` (the folder specified in `config`):

- transformed-1.zh: Transformed source embeddings that lie in the same semantic space as target embeddings.
- vocab.zh: Source language vocabulary.
- result.1: Translations of vocab.zh. For each source word, there will be at most 10 translations after the tab character, separated by space and sorted in decreasing order of the cosine similarity.

## Known Issue ##

It is recommended that the bleeding-edge version of Lasagne works with the latest development version of Theano. It has been tested on Lasagne version 0.2.dev1 and Theano version 0.9.0dev4.dev-RELEASE, but NaN may appear on Theano version 0.8.2.