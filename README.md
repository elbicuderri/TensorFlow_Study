# TensorFlow_Study

### 외우자 이 일곱줄
```python
for i, (img, label) in enumerate(train_loader):
    model_params = model.trainable_variables

    with tf.GradientTape() as tape: # torch는 forward하면 tensor에 autograd 된다.
        out = model(img)            # tf 는 tape에 기록하는 느낌으로 생각하면 된다.
        loss = loss_fn(out, label)

    grads = tape.gradient(loss, model_params)  # gradients 를 계산한다. loss.backward()
    optimizer.apply_gradients(zip(grads, model_params)) # optimizer.step()
```

### Conv2DTranspose
```python
import tensorflow as tf
from tensorflow.keras.layers import *

cifar = tf.random.uniform([1, 32, 32, 3], maxval=1)

out1 = Conv2DTranspose(filters=6,
                      kernel_size=3,
                      strides=1,
                      padding='valid')(cifar)

out2 = Conv2DTranspose(filters=6,
                      kernel_size=3,
                      strides=1,
                      padding='same')(cifar)

out3 = Conv2DTranspose(filters=6,
                      kernel_size=3,
                      strides=2,
                      padding='valid')(cifar)

out4 = Conv2DTranspose(filters=6,
                      kernel_size=3,
                      strides=2,
                      padding='same')(cifar)

print(out1.shape) # (1, 34, 34, 6)

print(out2.shape) # (1, 32, 32, 6)

print(out3.shape) # (1, 65, 65, 6)

print(out4.shape) # (1, 64, 64, 6)
```

[tf.function 공부](https://www.tensorflow.org/guide/function)

### tf static graph
```python
@tf.function
```

### tf.data 사용법

[Link 1](https://medium.com/trackin-datalabs/input-data-tf-data-%EC%9C%BC%EB%A1%9C-batch-%EB%A7%8C%EB%93%A4%EA%B8%B0-1c96f17c3696)

[Link 2](https://stackoverflow.com/questions/55627995/keras2-imagedatagenerator-or-tensorflow-tf-data)

[Link 3 - official](https://www.tensorflow.org/guide/data#preprocessing_data)

### load_weights()

```python
    checkpoint_path = "checkpoint/model_epoch_1.ckpt"

    model.load_weights(checkpoint_path) ## epoch 1 model

    checkpoint_dir = "checkpoint/"

    latest = tf.train.latest_checkpoint(checkpoint_dir)

    latest_model.load_weights(latest) 
```

### 마법의 한 줄
```python
tf.debugging.set_log_device_placement(True) # 무슨 일이 일어나는 지 보자
```

### tf.keras.layers.LSTM(ver 2.4.1)에 대한 설명
See the [Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn) for details about the usage of RNN API.<br>

Based on available runtime hardware and constraints, this layer will choose **different implementations**(cuDNN-based or pure-TensorFlow)<br>
to maximize the performance.<br>
If a GPU is available and all the arguments to the layer meet the requirement of the CuDNN kernel(see below for details),<br>
the layer will use a fast cuDNN implementation.<br>

The requirements to use the cuDNN implementation are:<br>

1. activation == tanh<br>
2. recurrent_activation == sigmoid<br>
3. recurrent_dropout == 0<br>
4. unroll is False<br>
5. use_bias is True<br>
6. Inputs, if use masking, are strictly right-padded.<br>
7. Eager execution is enabled in the outermost context.<br>



### tf 2.0에서 subclassing model API 사용시 BatchcNorm 과 Dropout 관리법
>
>
```python
class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        ...
        self.trainable = tf.constant(True) # 서치한 바로는 graph생성이 다시 안된다고 함. 속도 이득.
        self.training = tf.constant(True)

        self.batchnorm = BatchNormalization(trainable=self.trainable)
        self.dropout = Dropout(rate=self.rate, training=self.training)

for epoch in range(epochs):
    for i, (img, label) in enumerate(train_loader):
        model.trainable = tf.constant(True) # training 모드
        model.training = tf.constant(True) # training 모드
        
        model_params = model.trainable_variables

    for j, (val_img, val_label) in enumerate(valid_loader):
        model.trainable = tf.constant(False) # evaluating 모드
        model.training = tf.constant(False) # evaluating 모드
```

### tf.data.Dataset 사용법 (torch의 DataLoader와 비슷)
```python
dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

data_loader = dataset.map(preprocess).shuffle(60000, reshuffle_each_iteration=True).batch(32, drop_remainder=False)

#map : 전처리함수(data 하나에 대한 preprocess 함수를 작성하면 된다)
### 예를 들면 이렇게
def preprocess(x, y):
    image = tf.reshape(x, [32, 32, 3])
    image = tf.cast(image, tf.float32) / 255.0
    
    label = tf.one_hot(y, depth=10)
    label = tf.squeeze(label) # [1, 10] -> [10]
    return image, label
###
#shuffle : dataset 길이만큼 shuffle, reshuffle_each_iteration=False 면 같은 shuffle 반복
#batch: drop_remainder=True 면 마지막 batch_size 보다 작은 data 버림

for (x_batch, y_batch) in data_loader:
    ...
```

### torch 의 nn.Sequential 처럼 tf에서 block를 쌓는 방법..
```python
class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
    
        self.block = tf.keras.models.Sequential([ ... ]) # 이런 식으론 되지 않았다. 생각중.
        self.block = tf.keras.Model.Sequential([ ... ]) # 이것도 안되네요.. 그냥 함수로 짜야되나보네요..? 귀찮..


class Block(tf.keras.layers.Layer):   # 해결 완료. class로 선언해야됨. TF ver 올라가면서 위 두 방법도 사용가능한 듯.
    def __init__(self):
        super(Block, self).__init__()
        
    def call(self, x)
        
        return out

```

### Concatenate
```python
out = tf.keras.layers.Concatenate(axis=-1)([in1, in2]) # default: axis=-1
out = tf.keras.layers.concatenate([in1, in2])
```

### model.summary()
>
> subclassing API 방식 model을 만들면 model.summary() 가 안된다.
> 
```python
    def model(self):
        inputs = tf.keras.Input(shape=(32, 32, 3))
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

model = Model()

model.model().summary() 
# 해결완료. block으로 쌓인 부분 안 보임. 더 좋은 방법 찾아봅시다.
```

### control the randomness
```python
import tensorflow as tf
import os
import random
import numpy as np

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
def set_global_determinism(seed=SEED, fast_n_close=False):
    """
        Enable 100% reproducibility on operations related to tensor and randomness.
        Parameters:
        seed (int): seed value for global randomness
        fast_n_close (bool): whether to achieve efficient at the cost of determinism/reproducibility
    """
    set_seeds(seed=seed)
    if fast_n_close:
        return
    
    """
    logging.warning("*******************************************************************************")
    logging.warning("*** set_global_determinism is called,setting full determinism, will be slow ***")
    logging.warning("*******************************************************************************")
    """
    
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # https://www.tensorflow.org/api_docs/python/tf/config/threading/set_inter_op_parallelism_threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    from tfdeterminism import patch
    patch()
```


### einsum
```python
import tensorflow as tf 

# matmul
x = tf.random.uniform([2, 3])
y = tf.random.uniform([4, 3])

z = tf.einsum("ij, kj->ik", x, y)

print(z.shape)

#Fully Connected Layer
a = tf.random.uniform([32, 3, 228, 228])

b = tf.random.uniform([32, 228, 228, 3])

w1 = tf.random.uniform([10, 3 * 228 * 228])

w2 = tf.random.uniform([228 * 228 * 3, 10])

y1 = tf.einsum("nchw, kchw-> nk", a, w1.numpy().reshape([10, 3, 228, 228])) #PyTorch

y2 = tf.einsum("nhwc, hwck-> nk", b, w2.numpy().reshape([228, 228, 3, 10])) #TensorFlow

print(y1.shape)

print(y2.shape)
```


### tf의 data_format을 확인 하는 법
```python
import tensorflow.keras.backend as K
# default : channels_last

print(K.image_data_format())

K.set_image_data_format('channels_first')
print(K.image_data_format())

K.set_image_data_format('channels_last')
print(K.image_data_format())
```
