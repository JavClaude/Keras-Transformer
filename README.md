### Implementation of the Transformer architecture (Vaswani et. al. 2017) using Keras 

```python
import tensorflow as tf

from TransformerEncoder import TransformerEncoderLayer
from CLR import CyclicalLearningRate
from utils import create_padding_mask

tf.test.gpu_device_name()

I = tf.keras.layers.Input(shape=(512,))
M = create_padding_mask(I)
T = transformerEncoderLayer(num_encoder=4,
                            d_model=768,
                            num_heads=8, 
                            dff =1100, 
                            input_vocab_size=len(tokenizer.vocab), 
                            maximim_position_encoding=512, 
                            dropoutRate=0.1, 
                            pooling_activation="tanh", 
                            pooling_strategy="CLS", 
                            num_classes=2)(I, mask=M)

Transformer = tf.keras.Model(I, T)

#Important to set up a low LR for convergence
CyclicalLR = CyclicalLearningRate(min_lr=0.00001, max_lr=0.0001, stepsize=6000, cyclical_type="exp_range")

Transformer.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['acc'])

with tf.device('/device:GPU:0'):
    Transformer.fit(X_train, y_train, validation_split=0.1, callbacks=[CLR], batch_size=64, epochs=1)
```

## To Do

* Decoder Part
* BPE-Tokenizer
* requirements.txt
* test
