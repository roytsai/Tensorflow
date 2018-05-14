import tensorflow as tf
import numpy as np


#create data
# x_data = np.random.rand(10).astype(np.float32)
# y_data = x_data*0.1+0.3


x_data = [1.,2.,3.,4.]
y_data = [.4,.5,.6,.7]


print( x_data,y_data)
#create tensorflow neural network structure start
Weights = tf.Variable([1.0],tf.float32)
biases = tf.Variable([0.0],tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = Weights * x + biases
 
loss = tf.reduce_sum(tf.square(linear_model - y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
 
init = tf.global_variables_initializer()
#create tensorflow neural network structure end
 

 
session = tf.Session()
session.run(init)

for step in range(401):
    session.run(train,{x:x_data, y:y_data})
    if step%20 == 0:
        print(step, session.run(Weights),session.run(biases))
        
        
        
        