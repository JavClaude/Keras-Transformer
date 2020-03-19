import numpy as np

import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
    '''
    Description
    -----------
    Because there is no informations about relative position of each token
    It is necessary to inject some positional information

    Parameters
    ----------
        * input_vocab_size: Int, Size of vocabulary
        * output_dim: Int, Dimension of the dense embedding
        * d_model: Int, Dimension of the hidden state
        * maxposEncoding: Int, max position to encode
    '''
    def __init__(self,
                input_vocab_size=Int,
                output_dim=Int,
                d_model=Int,
                maxposEncoding=Int,
                embedding_initializer='uniform',
                activity_regularizer=None,
                **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        
        super(PositionalEncoding, self).__init__(**kwargs)
        self.input_vocab_size = input_vocab_size
        self.output_dim = output_dim
        self.d_model = d_model
        self.maxposEncoding = maxposEncoding
        self.Embedding = tf.keras.layers.Embedding(self.input_vocab_size, self.d_model)
        self.posEncoding = self._computePosition(self.maxposEncoding, self.d_model)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.built = True
        super(PositionalEncoding, self).build(input_shape)

    def call(self, x, mask):
        maxSeqLen = tf.shape(x)[1]

        x = self.Embedding(x)

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x += self.posEncoding[:, :maxSeqLen, :]

        return x
    
    def _getAngles(self, pos, i , d_model):
        '''
        Description
        -----------
        Add information about the relative position of each token in a sequence

        Paper: https://arxiv.org/abs/1706.03762

        Parameters
        ----------
            * pos: Int
            * i: Int, cos or sin
            * d_model: Int
        '''
        angles_rates = 1 / np.power(10000, (2 * (i//2)) /np.float32(d_model))
        return pos * angles_rates

    def _computePosition(self, position, d_model):
        '''
        Description
        -----------
        Add information about the relative position of each token in a sequence

        Paper: https://arxiv.org/abs/1706.03762

        Parameters
        ----------
            * position: Int
            * d_model: Int
        '''
        freqPos = self._getAngles(np.arange(position)[:, np.newaxis],
                                  np.arange(self.d_model)[np.newaxis, :],
                                  self.d_model)

        freqPos[:, 0::2] = np.sin(freqPos[:, 0::2])
        freqPos[:, 1::2] = np.cos(freqPos[:, 1::2])

        freqPos[np.newaxis, ...] #for batch dimension

        return tf.cast(freqPos, tf.float32)

        

