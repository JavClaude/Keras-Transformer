import tensorflow as tf

class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    '''
    Description
    -----------
    Core Layer of the Transformer Model

    Parameters
    ----------
        * d_Model: Int, Dim of the hidden state
        * num_heads: Int, number of Head Attention
        * returb_MHA: Bool, 
    '''
    def __init__(self,
                d_model,
                num_heads,
                return_MHA = False,
                kernel_initializer = 'glorot_normal',
                kernel_regularizer = None, 
                **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.d_model = d_model 
        self.num_heads = num_heads
        self.return_MHA = return_MHA
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        assert self.d_model % self.num_heads == 0, "Please, choose a dimension divisble by the total number of attention heads"

        self.depth = self.d_model // self.num_heads #Use // for integer division 

        self.wQuery = tf.keras.layers.Dense(self.d_model)
        self.wKey = tf.keras.layers.Dense(self.d_model)
        self.wValue = tf.keras.layers.Dense(self.d_model)

        self.unifyHeads = tf.keras.layers.Dense(self.d_model)
    
    def build(self, input_shape):
        assert len(input_shape) >= 2 # (batch_size, seq_len, d_model)
        self.built = True
        super(MultiHeadAttentionLayer, self).build(input_shape)

    def call(self, v, q, k, mask):
        # (batch_size, seq_len, d_model)
        batch_size = tf.shape(q)[0]

        input_size = tf.shape(q)[1]

        mixedQ = self.wQuery(q)
        mixedK = self.wQuery(k)
        mixedV = self.wValue(v)

        mixedQ = self._splitHeadAttention(mixedQ, batch_size, input_size)
        mixedK = self._splitHeadAttention(mixedK, batch_size, input_size)
        mixedV = self._splitHeadAttention(mixedV, batch_size, input_size)

    
    def _scaleDotProductAttention(self, v, q, k, mask):
        '''
        Description
        -----------
        The softmax function can be sensitive to very large input values:
            * kill the gradient
            * slow down / stop training 
        
        Parameters
        ----------
            * Q: Tensor
            * K: Tensor
            * V: Tensor
            * mask: mask before applying Softmax on zero padded input
        '''
        dotQueryKey = tf.matmul(q, k, transpose_b = True)

        dK = tf.cast(tf.shape(k)[-1], tf.float32)

        scaledAttentionLogits = dotQueryKey / tf.math.sqrt(dK)

        if mask is not None:
            scaledAttentionLogits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaledAttentionLogits, axis=-1) #Softmax on K

        tensorOut = tf.matmul(attention_weights, v)

        if return_MHA:

            return tensorOut, attention_weights
        
        return tensorOut

    
    def _splitHeadAttention(self, x, batch_size, input_size):
        '''
        Description
        -----------
        Define for each attention head its own dimension

        Parameters
        ----------
            * x: tf.Tensor, of shape (batch, seq, d_model)
            * batch_size:
            * input_size:
        
        Return
        ------
            * Tensor of shape (batch, head, seq, depth)
        '''
        x = tf.reshape(x, (batch_size, input_size, self.num_heads, self.depth))

        return tf.transpose(x, perm = [0, 2, 1, 3]) # (batch, head, seq, depth)
    


