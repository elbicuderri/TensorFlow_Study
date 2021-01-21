import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical

iris_data = load_iris()

x = iris_data.data
y = iris_data.target
y = to_categorical(y)

with open("iris_data.bin", "wb") as f_data:
    x_array = np.array(x, dtype=np.float32)

    print(x_array.shape)

    x_array.tofile(f_data)

with open("iris_label.bin", "wb") as f_label:

    y_array = np.array(y, dtype=np.float64)

    print(y_array.shape)

    y_array.tofile(f_label)