import tensorflow as tf
import numpy as np

# load_model: tensorflow 모델을 laod하기 위한 함수
from tensorflow.keras.models import load_model

# model 객체를 불러온다.
model = load_model("tf_iris_model.h5")

# weights: model 모든 weight(numpy.ndarray)를 list형태로 불러온다. type: list[numpy.ndarray]
weights = model.get_weights()

with open("tf_iris_weight.bin", "wb") as f:
    # 처음 40bytes는 float32(0) * 10개를 채워준다.
    (np.asarray([0 for _ in range(10)], dtype=np.float32)).tofile(f)

    # list 에서 numpy.ndarray를 하나씩 꺼내서 binary 형식으로 저장한다.
    for weight in weights:
        print(weight.shape, isinstance(weight, np.ndarray))
        weight.tofile(f)
