# TensorFlow_Study

**외우자 이 다섯줄**
```python
    for i, (img, label) in enumerate(train_loader):
        model_params = model.trainable_variables
        with tf.GradientTape() as tape: # torch 는 forward를 하면 자동으로 jacobian matrix 을 생성해 autograd를 위해 준비 
            out = model(img)            # tf 는 tape라는 것으로 감싸놓는 느낌적인 느낌으로 생각하면 된다.
            loss = loss_fn(out, label)
        grads = tape.gradient(loss, model_params)  # gradients 를 계산한다. loss.backward()
        optimizer.apply_gradients(zip(grads, model_params)) # optimizer.step()
```

**마법의 한 줄**
```python
tf.debugging.set_log_device_placement(True) # 무슨 일이 일어나는 지 보자
```

> 오늘의 과제
> ~~TF는 bn 과 dropout을 어떻게 관리하지?~~
>
> 해결 완료
>
> like this
```python
        self.trainable = True
        self.batchnorm = tf.keras.layers.BatchNormalization(trainable=self.trainable)
        self.dropout = tf.keras.layers.Dropout(rate=self.rate, training=self.trainable) 
        
        for epoch in range(epochs):
            for i, (img, label) in enumerate(train_loader):
                model.trainable = True
                model_params = model.trainable_variables

            for j, (val_img, val_label) in enumerate(valid_loader):
                model.trainable = False            
# 아마도 해결? @, @...
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


```python
# tf.data.Dataset 사용법 (torch의 DataLoader와 비슷)

dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

data_loader = dataset.map(preprocess).shuffle(60000, reshuffle_each_iteration=True).repeat(3).batch(32, drop_remainder=False)

#map : 전처리함수(data 하나에 대한 preprocess 함수를 작성하면 된다)
### 예를 들면 이렇게
def preprocess(x, y):
    x = tf.reshape(x, [32, 32, 3])
    image = tf.cast(x, tf.float32) / 255.0
    label = tf.one_hot(y, depth=10)
    label = tf.squeeze(label)
    return image, label
###
#shuffle : dataset 길이만큼 shuffle, reshuffle_each_iteration=False 면 같은 shuffle 반복
#repeat : epoch만큼 반복된 dataset 생성
#batch: drop_remainder=True 면 마지막 batch_size 보다 작은 data 버림

for (x_batch, y_batch) in data_loader:
    ...
```

```python
class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
    
        self.block = tf.keras.models.Sequential([ ... ]) # 이런 식으론 되지 않았다. 생각중.
```

```python
out = concatenate([in1, in2])
out = in1 + in2 # in1, in2 가 tf.Tensor 일 경우
```

**einsum is all you need**
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
