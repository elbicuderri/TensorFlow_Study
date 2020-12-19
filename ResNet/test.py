import tensorflow as tf 

a = tf.constant([[1, 2]], dtype=tf.float32)
b = tf.constant([[3, 4]], dtype=tf.float32)

c = a + b
print(c)