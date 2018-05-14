import tensorflow as tf
import numpy as np

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
import os
import sys

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

#create data
# x_data = np.random.rand(10).astype(np.float32)
# y_data = x_data*0.1+0.3


x_data = [1.,2.,3.,4.]
y_data = [.4,.5,.6,.7]


# print( x_data,y_data)
# #create tensorflow neural network structure start
# Weights = tf.Variable([1.0],tf.float32)
# biases = tf.Variable([0.0],tf.float32)
# x = tf.placeholder(tf.float32)
# y = tf.placeholder(tf.float32)
# linear_model = Weights * x + biases
#  
# loss = tf.reduce_sum(tf.square(linear_model - y))
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
#  
#  
# session = tf.Session()
# session.run(tf.global_variables_initializer())
# 
# for step in range(401):
#     session.run(train,{x:x_data, y:y_data})
#     if step%20 == 0:
#         print(step, session.run(Weights),session.run(biases))
        

g_2 = tf.Graph()
with g_2.as_default():
    x_2 = tf.placeholder(tf.float32, [None, 1], name="input")
    W_2 = tf.constant([[2.0]], name="constant_W")
    b_2 = tf.constant( [2.0], name="constant_b")
    y_2 = tf.add(tf.matmul(x_2, W_2),b_2,name="output")


    sess_2 = tf.Session()
    
    sess_2.run(tf.global_variables_initializer())
    
    print(sess_2.run(x_2, feed_dict={x_2:[[5.0]]}))
    print(sess_2.run(W_2))
    print(sess_2.run(b_2))
    print(sess_2.run(y_2, feed_dict={x_2:[[5.0]]}))
    
    graph_def = g_2.as_graph_def()
     
    tf.train.write_graph(graph_def, './model/beginner-export',
                                     'beginner-graph-linear.pb', as_text=False)
    sess_2.close()

        
        