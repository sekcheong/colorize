import tensorflow as tf
import numpy as np
# Creates a graph.
y1 = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
y2 = tf.constant([1.2, 1.0, 3.1, 2.8, 3.0, -3.0, 3.0, 5.0])
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
u1, v1 = tf.split(y1, 2)
u2, v2 = tf.split(y2, 2)

d1 = tf.square(tf.subtract(u1, u2))
d2 = tf.square(tf.subtract(v1, v2))

d = tf.sqrt( d1 + d2 )
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Runs the op.
# print (sess.run(u1))
# print (sess.run(u2))
# print (sess.run(v1))
# print (sess.run(v2))
#z = tf.placeholder(tf.float32, shape=(1024, 1024))
z = tf.placeholder(tf.float32, shape=(1, 1))
values = np.array([1.0], dtype=np.float32)
tf.assign(z, values)
print (sess.run(d1))
print (sess.run(d2))
print (sess.run(d))
