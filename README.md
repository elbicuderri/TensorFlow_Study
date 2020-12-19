# TensorFlow_Study


20201219

- tf.data.Dataset 사용법 (torch의 DataLoader와 비슷)

dataset = tf.data.Dataset((x_data, y_data)))

data_loader = dataset.batch(batch_size)  ### 여기에 shuffle, map, repeat 같은 다른 옵션이 있다. 

for (x_batch, y_batch) in data_loader:
  
```python
class Model(tf.keras.Model):
  def __init__(self):
    super().__init__()
    
    self.block = tf.keras.model.Sequential([ ... ]) # 이런 식으론 되지 않았다.
```
```python
out = concatenate([in1, in2])
out = in1 + in2 # in1, in2 가 tf.Tensor 일 경우
```
