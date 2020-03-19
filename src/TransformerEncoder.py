import tensorflow as tf
from . import PositionalEncodingLayer, MultiHeadAttentionLayer, EncoderLayer, EncoderPoolerLayer

class TransformerEncoder(tf.keras.layers.Layer):
    '''
    Description
    -----------
        
    Return
    ------

    '''
    def __init__(self,
                 num_encoder = int,
                 d_model = int,
                 num_heads = int,
                 dff = int,
                 input_vocab_size = int,
                 max_pos_encoding = int,
                 dropoutRate = int,
                 pooling_activation = str,
                 pooling_strategy = str,
                 n_classes = int,
                 **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.nlayers = self.num_encoder
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff

        self.input_vocab_size = input_vocab_size
        self.max_pos_encoding = max_pos_encoding

        self.pooling_activation = pooling_activation
        self.pooling_strategy = pooling_strategy
        self.n_classes = n_classes
        self.dropoutRate = dropoutRate


        self.Embedding = PositionalEncodingLayer(self.input_vocab_size, self.d_model, self.max_pos_encoding)

        self.encoderLayers = [EncoderLayer(self.d_model, self.num_heads, self.dff, False, self.dropoutRate)]

        self.encoderPooler = EncoderPoolerLayer(self.d_model, self.pooling_activation, self.pooling_strategy)

        if self.n_classes > 1:
            self.logits = tf.keras.layers.Dense(n_classes, activation="softmax")
        
        else:
            self.logits = tf.keras.layers.Dense(n_classes, activation="sigmoid")
        
    def call(self, x, mask = None):
        x = self.Embedding(x)

        for i in list(range(self.nlayers)):
            x = self.encoderLayers[i](x, mask)
        
        x = self.encoderPooler(x)

        x = self.logits(x)

        return x