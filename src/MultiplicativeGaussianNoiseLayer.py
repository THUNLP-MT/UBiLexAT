import theano
import theano.tensor as T

from lasagne.layers.base import Layer
from lasagne.random import get_rng

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class MultiplicativeGaussianNoiseLayer(Layer):
	def __init__(self, incoming, sigma=0.1, **kwargs):
		super(MultiplicativeGaussianNoiseLayer, self).__init__(incoming, **kwargs)
		self._srng = RandomStreams(get_rng().randint(1, 2147462579))
		self.sigma = sigma

	def get_output_for(self, input, deterministic=False, **kwargs):
		"""
		Parameters
		----------
		input : tensor
			output from the previous layer
		deterministic : bool
			If true noise is disabled, see notes
		"""
		if deterministic or self.sigma == 0:
			return input
		else:
			return input * self._srng.normal(input.shape,
											 avg=1.0,
											 std=self.sigma)
