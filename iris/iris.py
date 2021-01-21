import tensorflow as tf

"""
Sequential 형식을 이용한다. 
Dense: Dense layer
to_categorical: one-hot encoding을 해주는 함수
"""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# skleran에서 제공해주는 iris datasets
from sklearn.datasets import load_iris

# define the model
model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu'))
model.add(Dense(3, activation='softmax'))

# summary the model
model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )

# load IRIS dataset
iris_data = load_iris()

x = iris_data.data
y = iris_data.target
y = to_categorical(y)

model.fit(x, y, batch_size=1, epochs=1000, validation_split=0.1, shuffle=True)

model.save("tf_iris_model.h5")
