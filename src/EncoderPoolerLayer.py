from . import MultiHeadAttentionLayer
from . import PositionalEncodingLayer
import tensorflow as tf

class EncoderPoolerLayer(tf.keras.layers.Layer):
    '''
    Description
    -----------
    Pool the 3DTensor to 2DTensor

    Parameters:
    -----------
        *

    Return
    ------

    '''
    def __init__(self = int,
                 d_model = int,
                 pooling_activation = 'tanh',
                 pooling_strategy = str,
                 **kwargs):
        super(EncoderPoolerLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.pooling_activation = pooling_activation
        self.pooling_strategy = pooling_strategy

        if self.pooling_strategy == "Average":
            self.poolingLayer = tf.keras.layers.GlobalAveragePooling1D

        elif self.pooling_strategy == "CLS":
            self.poolingLayer = tf.keras.layers.Dense(self.d_model,
                                                      activation = self.pooling_activation)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.built = True
        super(EncoderPoolerLayer, self).build(input_shape)
    
    def call(self, x):
        if self.pooling_strategy == "Average":
            x = self.poolingLayer(x)
        
        elif self.pooling_strategy == "CLS":
            x = self.poolingLayer(x[:,0])

        return x