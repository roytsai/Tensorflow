import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


def add_layer(inputs, in_size, out_size, activation_function = None):
    
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs, Weights)+biases
    
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)    
    return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_prediction = session.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_prediction,1), tf.argmax(v_ys,1))
    #tf.argmax : 以axis為基準，回傳該行中最大的數值的位置，比方
    #tf.argmax([[1,2,3],[3,4,2]], axis=0) = [1 1 0]
    #tf.argmax([[1,2,3],[3,4,2]], axis=1) = [2 1]
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = session.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result

# Import data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


#define placeholder for input to n
xs = tf.placeholder(tf.float32, [None,784])
ys = tf.placeholder(tf.float32, [None,10])


#add output layer
prediction = add_layer(xs, 784, 10, activation_function = tf.nn.softmax)
#softmax 這個激勵函數通常是用來做classifcation 

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))    
train_step = tf.train.AdamOptimizer(0.004).minimize(cross_entropy)


# google 的代碼
# cross_entropy = tf.reduce_mean(
#       tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

session = tf.Session()
session.run(tf.global_variables_initializer())

#train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
    if i%50 == 0:
        #print(compute_accuracy(mnist.test.images, mnist.test.labels))
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
        #取得每一行最大的值的位置，也就是得知預測出來的數字
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #tf.cast : 型態轉換 true-> 1
        #tf.reduce_mean : 全部數字相加除以全部元素個數，也就是我們要的百分比
        print(session.run(accuracy, feed_dict={xs: mnist.test.images,
                                      ys: mnist.test.labels}))



