# TensorFlow_Study


```python
# tf.data.Dataset 사용법 (torch의 DataLoader와 비슷)

dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

data_loader = dataset.map(preprocess).shuffle(60000, reshuffle_each_iteration=True).repeat(3).batch(32, drop_remainder=False)

#preproces : 전처리함수
#shuffle : dataset 길이만큼 shuffle, reshuffle_each_iteration=False 같은 shuffle 반복
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
