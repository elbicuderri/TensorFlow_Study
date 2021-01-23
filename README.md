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
See the [Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn) for details about the usage of RNN API.

Based on available runtime hardware and constraints, this layer will choose different implementations(cuDNN-based or pure-TensorFlow)<br>
to maximize the performance.<br>
If a GPU is available and all the arguments to the layer meet the requirement of the CuDNN kernel(see below for details), 
the layer will use a fast cuDNN implementation.<br>

The requirements to use the cuDNN implementation are:

activation == tanh
recurrent_activation == sigmoid
recurrent_dropout == 0
unroll is False
use_bias is True
Inputs, if use masking, are strictly right-padded.
Eager execution is enabled in the outermost context.



### tf는 bn 과 dropout을 어떻게 관리하지?(custom train일 때...)
>
> like this... 아마도?
>
```python
self.trainable = True
self.trainable = tf.constant(True) # ? 이런게 되는 지.. 확인 필요. 서치한 바로는 graph생성이 다시 안된다고 함. 속도 이득.

# 참고: [tf.function](https://www.tensorflow.org/guide/function)

self.training = True
self.training = tf.constant(True) # ? 이런게 되는 지.. 확인 필요

self.batchnorm = BatchNormalization(trainable=self.trainable)
self.dropout = tf.keras.layers.Dropout(rate=self.rate, training=self.training)

for epoch in range(epochs):
    for i, (img, label) in enumerate(train_loader):
        model.trainable = True
        self.trainable = tf.constant(True)
        
        model.training = True
        self.training = tf.constant(True)
        
        model_params = model.trainable_variables

    for j, (val_img, val_label) in enumerate(valid_loader):
        model.trainable = False
        model.trainable = tf.constant(False)
        
        model.training = False
        model.training = tf.constant(False)
# 아마도 해결? batchnorm은 따로 할 필요 없고 dropout만 train, infer시 바꾸면 된다.
# 아니다 model.trainable = False 설정해야 한다. resnet/tf_resnet_updated.py 참고
# 핸즈온 머신러닝 2판 p494 "가장 중요한 것은 이 훈련 반복이 훈련과 테스트 시에 
# 다르게 동작하는 층(예를 들면 BatchNormalizatation이나 Dropout)을 
# 다루지 못한다는 점입니다. 이를 처리하려면 training=True로 모델을 호출하여 
# 필요한 모든 층에 이 매개변수가 전파되도록 해야 합니다." - 여길 보면 저렇게 하면 맞는 거 같긴 한데...
```
The meaning of setting layer.trainable = False is to freeze the layer, 
i.e. its internal state will not change during training: its trainable weights will not be updated during fit() or train_on_batch(), and its state updates will not be run.

Usually, this does not necessarily mean that the layer is run in inference mode (which is normally controlled by the training argument that can be passed when calling a layer). "Frozen state" and "inference mode" are two separate concepts.

However, in the case of the BatchNormalization layer, setting trainable = False on the layer means that the layer will be subsequently run in inference mode (meaning that it will use the moving mean and the moving variance to normalize the current batch, rather than using the mean and variance of the current batch).

This behavior has been introduced in TensorFlow 2.0, in order to enable layer.trainable = False to produce the most commonly expected behavior in the convnet fine-tuning use case.

Note that:
This behavior only occurs as of TensorFlow 2.0. In 1.*, setting layer.trainable = False would freeze the layer but would not switch it to inference mode.
Setting trainable on an model containing other layers will recursively set the trainable value of all inner layers.
If the value of the trainable attribute is changed after calling compile() on a model, the new value doesn't take effect for this model until compile() is called again.

Reference:
Ioffe and Szegedy, 2015.

### tf.data.Dataset 사용법 (torch의 DataLoader와 비슷)
```python
dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

data_loader = dataset.map(preprocess).shuffle(60000, reshuffle_each_iteration=True).batch(32, drop_remainder=False)

#map : 전처리함수(data 하나에 대한 preprocess 함수를 작성하면 된다)
### 예를 들면 이렇게
def preprocess(x, y):
    x = tf.reshape(x, [32, 32, 3])
    image = tf.cast(x, tf.float32) / 255.0
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


class Block(tf.keras.layers.Layer):   # 해결 완료. class로 선언해야됨.
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
> 대충 방법은 알 거 같은데 귀찮다..
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


### einsum is all you need
>
> tensorflow도 einsum을 지원한다. numpy도 지원한다. 앞으로 연산할때 shape 생각하기 쉬워질듯
>

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


### tf의 data_format의 확인 하는 법
```python
import tensorflow.keras.backend as K
# default : channels_last
print(K.image_data_format())
K.set_image_data_format('channels_first')
print(K.image_data_format())
K.set_image_data_format('channels_last')
print(K.image_data_format())
```
