# TensorFlow_Study

> 오늘의 과제
> TF는 bn 과 dropout을 어떻게 관리하지?
>
>
```python
self.batchnorm = BatchNormalization(trainable=True)
...
model = Model(...)
...
model.batchorm.trainable = False 
## somthing like this? 확인 필요
## 저렇게 하면 bn의 mean과 var는 어떤걸 사용하는 걸까
## dropout 은 training 이란 키워드가 있는데 
## 훈련 중 val_loss 체크를 할때 어떻게 되는 걸까..
## 은제 확인하냐
```

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
