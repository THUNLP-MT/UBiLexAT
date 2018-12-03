import sys
import time
import numpy as np
import cPickle
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
import theano
import theano.tensor as T
import lasagne

from embeddings import WordEmbeddings
from MultiplicativeGaussianNoiseLayer import MultiplicativeGaussianNoiseLayer

def save_model():
	params_vals = lasagne.layers.get_all_param_values([discriminator.l_out, gen_l_out])
	cPickle.dump(params_vals, open(MODEL_FILENAME, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

def load_model():
	params = lasagne.layers.get_all_params([discriminator.l_out, gen_l_out])
	params_vals = cPickle.load(open(MODEL_FILENAME, 'rb'))
	for i, param_val in enumerate(params_vals):
		params[i].set_value(param_val)

def cosine_sim(a_mat, b_mat):
	dp = (a_mat * b_mat).sum(axis=1)
	a_norm = a_mat.norm(2, axis=1)
	b_norm = b_mat.norm(2, axis=1)
	return dp / (a_norm * b_norm)

class Discriminator(object):
	def __init__(self, input_dim=40, depth=1, width=40, input_noise_param=.2, hidden_noise_param=.5, learning_rate=0.001):
		print >> sys.stderr, 'Building computation graph for discriminator...'
		self.input_var = T.matrix('input')
		self.target_var = T.matrix('target')

		self.l_out = self.buildFeedForward(self.input_var, input_dim, depth, width, input_noise_param, hidden_noise_param)

		self.prediction = lasagne.layers.get_output(self.l_out)
		self.loss = lasagne.objectives.binary_crossentropy(self.prediction, self.target_var).mean()
		self.accuracy = T.eq(T.ge(self.prediction, 0.5), self.target_var).mean()

		self.params = lasagne.layers.get_all_params(self.l_out, trainable=True)
		self.updates = lasagne.updates.adam(self.loss, self.params, learning_rate=learning_rate)

		print >> sys.stderr, 'Compiling discriminator...'
		self.train_fn = theano.function([self.input_var, self.target_var], [self.loss, self.accuracy], updates=self.updates)
	
	def buildFeedForward(self, input_var=None, input_dim=40, depth=1, width=40, input_noise_param=.2, hidden_noise_param=.5):
		# Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
		network = lasagne.layers.InputLayer(shape=(None, input_dim),
											input_var=input_var)
		if input_noise_param:
			network = MultiplicativeGaussianNoiseLayer(network, input_noise_param) if args.input_noise == 'gaussian' else lasagne.layers.dropout(network, input_noise_param)
		# Hidden layers and dropout:
		nonlin = lasagne.nonlinearities.rectify
		for _ in range(depth):
			network = lasagne.layers.DenseLayer(
					network, width, nonlinearity=nonlin)
			if hidden_noise_param:
				network = MultiplicativeGaussianNoiseLayer(network, hidden_noise_param) if args.hidden_noise == 'gaussian' else lasagne.layers.dropout(network, hidden_noise_param)
		# Output layer:
		network = lasagne.layers.DenseLayer(network, 1, nonlinearity=lasagne.nonlinearities.sigmoid)
		return network

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('config', help='Directory name.')
parser.add_argument('lang1', help='Source language.')
parser.add_argument('lang2', help='Target language.')
parser.add_argument('--Dlayers', type=int, default=1, help='Number of hidden layers of D.')
parser.add_argument('--Ddim', type=int, default=500, help='Dimensionality of hidden layers of D.')
parser.add_argument('--input-noise', choices=['gaussian', 'dropout'], default='gaussian', help='D input noise type.')
parser.add_argument('--hidden-noise', choices=['gaussian', 'dropout'], default='gaussian', help='D hidden noise type.')
parser.add_argument('--input-noise-param', type=float, default=0.5, help='Gaussian standard deviation, or dropout probability.')
parser.add_argument('--hidden-noise-param', type=float, default=0.5, help='Gaussian standard deviation, or dropout probability.')
parser.add_argument('--alt-loss', action='store_true', help='Use -log(D) instead of log(1-D).')
parser.add_argument('--recon-weight', type=float, default=0, help='Reconstruction term weight.')
parser.add_argument('--num-minibatches', type=int, default=1000000, help='Number of minibatches.')
parser.add_argument('--num-save', type=int, default=0, 
				help='If > 0, indicates the number of models to save. Otherwise, save based on G loss.')
args = parser.parse_args()
	
DISCR_NUM_HIDDEN_LAYERS = args.Dlayers
DISCR_HIDDEN_DIM = args.Ddim
HALF_BATCH_SIZE = 128

MODEL_FILENAME = 'model.pkl'

rng = check_random_state(0)

lang1 = args.lang1
lang2 = args.lang2
dataDir = 'data/' + args.config + '/'

print >> sys.stderr, 'Loading', lang1, 'embeddings...'
we1 = WordEmbeddings()
we1.load_from_word2vec(dataDir, lang1)
we1.downsample_frequent_words()
we1.vectors = normalize(we1.vectors).astype(theano.config.floatX)
we_batches1 = we1.sample_batches(batch_size=HALF_BATCH_SIZE, random_state=rng)

print >> sys.stderr, 'Loading', lang2, 'embeddings...'
we2 = WordEmbeddings()
we2.load_from_word2vec(dataDir, lang2)
we2.downsample_frequent_words()
we2.vectors = normalize(we2.vectors).astype(theano.config.floatX)
we_batches2 = we2.sample_batches(batch_size=HALF_BATCH_SIZE, random_state=rng)

assert we1.embedding_dim == we2.embedding_dim
d = we1.embedding_dim

discriminator = Discriminator(d, DISCR_NUM_HIDDEN_LAYERS, DISCR_HIDDEN_DIM, args.input_noise_param, args.hidden_noise_param)

print >> sys.stderr, 'Building computation graph for generator...'

gen_input_var = T.matrix('gen_input_var')

gen_l_in = lasagne.layers.InputLayer(shape=(None, d), input_var=gen_input_var, name='gen_l_in')
gen_l_out = lasagne.layers.DenseLayer(gen_l_in, num_units=d, nonlinearity=None, W=lasagne.init.Orthogonal(), b=None, name='gen_l_out')

generation = lasagne.layers.get_output(gen_l_out)
generation.name = 'generation'

discriminator_prediction = lasagne.layers.get_output(discriminator.l_out, generation, deterministic=True)
adv_gen_loss = -T.log(discriminator_prediction).mean() if args.alt_loss else T.log(1.0 - discriminator_prediction).mean()
adv_gen_loss.name = 'adv_gen_loss'

dec_l_out = lasagne.layers.DenseLayer(gen_l_out, num_units=d, nonlinearity=None, W=gen_l_out.W.T, b=None, name='dec_l_out')

reconstruction = lasagne.layers.get_output(dec_l_out)
reconstruction.name = 'reconstruction'
recon_gen_loss = 1.0 - cosine_sim(gen_input_var, reconstruction).mean()
recon_gen_loss.name = 'recon_gen_loss'

if args.recon_weight == 0:
	gen_loss = adv_gen_loss
else:
	gen_loss = adv_gen_loss + args.recon_weight * recon_gen_loss
gen_loss.name = 'gen_loss'

gen_params = lasagne.layers.get_all_params(dec_l_out, trainable=True)
gen_updates = lasagne.updates.adam(gen_loss, gen_params, learning_rate=0.001)

grad_norm = T.grad(adv_gen_loss, gen_l_out.W).norm(2, axis=1).mean()

print >> sys.stderr, 'Compiling generator...'
gen_train_fn = theano.function([gen_input_var], [gen_loss, recon_gen_loss, adv_gen_loss, generation, grad_norm], updates=gen_updates)


print >> sys.stderr, 'Training...'
print_every_n = 100
numBatches = args.num_minibatches
gloss_min = 10000000
modelID = 1
X = np.zeros((2 * HALF_BATCH_SIZE, d), dtype=theano.config.floatX)
target_mat = np.vstack([np.zeros((HALF_BATCH_SIZE, 1)), np.ones((HALF_BATCH_SIZE, 1))]).astype(theano.config.floatX)
start_time = time.time()
print >> sys.stderr, 'Initial det(W)', np.linalg.det(gen_l_out.W.get_value())
for batch_id in xrange(1, numBatches + 1):
	id1 = next(we_batches1)
	id2 = next(we_batches2)
	X[:HALF_BATCH_SIZE] = we1.vectors[id1]

	# Generator
	gen_loss_val, recon_gen_loss_val, adv_gen_loss_val, X_gen, grad_norm_val = gen_train_fn(X[:HALF_BATCH_SIZE])

	# Discriminator
	X[:HALF_BATCH_SIZE] = X_gen
	X[HALF_BATCH_SIZE:] = we2.vectors[id2]
	loss_val, accuracy_val = discriminator.train_fn(X, target_mat)

	if batch_id % print_every_n == 0:
		W = gen_l_out.W.get_value()
		print '%s %s %s %s %s %s %s %s' % (batch_id, accuracy_val, loss_val, gen_loss_val, recon_gen_loss_val, adv_gen_loss_val, grad_norm_val, np.linalg.norm(np.dot(W.T, W) - np.identity(d)))
	
	if args.num_save > 0:
		if batch_id % (numBatches / args.num_save) == 0:
			save_model()
			W = gen_l_out.W.get_value()
			print >> sys.stderr, 'recon_gen_loss_val', recon_gen_loss_val, '||W^T*W - I||', np.linalg.norm(np.dot(W.T, W) - np.identity(d)), 'det(W)', np.linalg.det(W)
			we1.transformed_vectors = np.dot(we1.vectors, W)
			we1.save_transformed_vectors(dataDir + 'transformed-' + str(modelID) + '.' + lang1)
			modelID += 1
	else:
	 	if batch_id > 10000 and gen_loss_val < gloss_min:
	 		gloss_min = gen_loss_val
	 		print >> sys.stderr, batch_id, gloss_min
			save_model()
			W = gen_l_out.W.get_value()
			print >> sys.stderr, 'recon_gen_loss_val', recon_gen_loss_val, '||W^T*W - I||', np.linalg.norm(np.dot(W.T, W) - np.identity(d)), 'det(W)', np.linalg.det(W)
			we1.transformed_vectors = np.dot(we1.vectors, W)
			we1.save_transformed_vectors(dataDir + 'transformed-' + str(modelID) + '.' + lang1)
print >> sys.stderr, (time.time() - start_time) / 60, 'min'
