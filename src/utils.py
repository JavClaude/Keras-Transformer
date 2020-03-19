import tensorflow as tf

def create_padding_mask(inputs):
    '''
    Description
    -----------
    Mask zero padding element

    Parameters
    ----------
        * inputs: Array like, sequence of tokens
    Return
    ------
    '''
    seq = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]