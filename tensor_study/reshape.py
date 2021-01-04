import tensorflow as tf

# matmul
x = tf.random.uniform([2, 3])
y = tf.random.uniform([4, 3])

print(x)

print(y)

x = x.numpy().reshape(-1) # add numpy()

print(x)