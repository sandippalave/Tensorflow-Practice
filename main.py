import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
x = tf.constant(4, shape=(1,1))

x = tf.ones((3,3))
x = tf.zeros(3,3)
x = tf.eye(3)        # Identity matrix
x = tf.random.normal((3,3), mean=0, stddev=1)
x = tf.random.uniform((1,3), minval=0, maxval=1)
x = tf.range(1,10,2)
x = tf.cast(x, dtype = tf.float64)

# Mathematical Operations
x = tf.constant([1,2,3])
y = tf.constant([4,2,3])
z = tf.add(x,y)     # or z = x + y

z = tf.tensordot(x,y, axes=1)
z = tf.reduce_sum(x*y,axis=0)


# Indexing
x = tf.constant([0,1,1,2,3,1,2,3])
x = tf.constant([[1,2],
                 [3,4],
                 [5,4]])
print(x[0,:])

# Reshaping
x = tf.range(9)
print(x)

x = tf.reshape(x, (3,3))
print(x)

x = tf.transpose(x, perm=[1,0])
print(x)
