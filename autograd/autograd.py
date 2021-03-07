import tensorflow as tf

a, y = tf.constant(2.0), tf.constant(8.0)
x = tf.Variable(10.0) # In practice, we start with a random value
# loss = tf.math.abs(a * x - y)

# UDF for training
def train_func():
        
    with tf.GradientTape() as tape:
        loss = tf.math.abs(a * x - y)

    # calculate gradient
    dx =  tape.gradient(loss, x)
    print(f"x = {x.numpy()}, dx = {dx:.2f}")

    # update x <- x - dx
    x.assign(x - dx)


# Run train_func() UDF repeatelt
for i in range(4):
    train_func()