

import tensorflow as tf
import numpy as np





inputData = tf.placeholder(tf.float32, [1, 3], name = "x")
Weights = tf.Variable(tf.random_normal([3, 10]))
biases = tf.Variable(tf.random_normal([1, 10]))

y = tf.matmul(inputData , Weights)+ biases


session = tf.Session()
session.run(tf.global_variables_initializer())
print(session.run(y, feed_dict={inputData:[[1.0,2.0,3.0]]}))
session.close()
 

