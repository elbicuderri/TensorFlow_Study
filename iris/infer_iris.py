# 실행시 인자를 전달하기 위한 library
import argparse

import tensorflow as tf
from tensorflow.keras.models import Model, load_model

from sklearn.datasets import load_iris

parser = argparse.ArgumentParser(description='Display the logits of the model')
"""
기본 인자로 3을 설정
150개를 다 보고 싶으면
python infer_iris.py -n 150
"""
parser.add_argument('-n', type=int, help='how many display, default=3', default=3)

args = parser.parse_args()

batch = args.n

# 150이 넘으면 에러 발생
assert (batch <= 150), "not over 150"

iris_data = load_iris()

x = iris_data.data

model = load_model("tf_iris_model.h5")

x = x[:batch, :]

print("Data Ready")

output = model.predict(x)

print(output.shape)

print("\n")

for i in range(batch):
    print(f"{i}th logit: ")
    print(output[i, :])
    print("")