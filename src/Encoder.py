from MultiHeadAttention import MultiHeadAttentionLayer
import tensorflow as tf

class EncoderLayer(tf.keras.layers.Layer):
    '''
    Description
    -----------
    EncoderLayer for the transformer architecture:
        * MHA
        * LayerNorm + Residual Connection
        * FFN
        * Dropout
        * LayerNorm + Residual Connection
        
    Return
    ------

    '''
    def __init__(self,
                 d_model = int,
                 num_heads = int,
                 dff = int,
                 return_attention = False,
                 dropoutRate = int,
                 **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.return_attention = return_attention
        self.dropoutRate = dropoutRate
        self.MHA = MultiHeadAttentionLayer(self.d_model, self.num_heads, )
        self.FFN = self._feedForwardNetwork(self.d_model, self.dff)

        self.LayerNorm1 = tf.keras.layers.LayerNormalization()
        self.LayerNorm2 = tf.keras.layers.LayerNormalization()

        self.DropOut1 = tf.keras.layers.Dropout(self.dropoutRate)
        self.DropOut2 = tf.keras.layers.Dropout(self.dropoutRate)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.built = True
        super(EncoderLayer, self).build(input_shape)
    
    def call(self, x, mask):
        if self.return_attention == False:
             attentionOutput = self.MHA(x, x, x, mask)
        
        else:
            attentionOutput, _ = self.MHA(x, x, x, mask)
        
        attentionOutput = self.DropOut1(attentionOutput)
        firstOutput = self.LayerNorm1(x + attentionOutput)

        feedForwardOutput = self.FFN(firstOutput)
        feedForwardOutput = self.DropOut2(feedForwardOutput)

        secondOutput = self.LayerNorm2(firstOutput + feedForwardOutput)

        return secondOutput

    def _feedForwardNetwork(self, d_model, dff):
        return tf.keras.Sequential([tf.keras.layers.Dense(self.dff, activation='relu'), tf.keras.layers.Dense(self.d_model)])
