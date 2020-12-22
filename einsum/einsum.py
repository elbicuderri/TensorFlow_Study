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