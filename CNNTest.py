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
    y_pre = session.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = session.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result



def weight_variable(shape):
    #tf.random_normal 跟truncated_normal的差別是，truncated_normal會剪裁掉兩個標準差外的值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    #bias用正值較好
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, filter):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, filter, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    # Must have ksize[0] = ksize[3] = 1
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Import data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 28*28])/255.   # [batch_size ,28x28]
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28, deep(=1)]

## conv1 layer ##

W_conv1 = weight_variable([5,5, 1, 32]) # patch 5x5, deep in size 1, deep out size 32 (最後一個數字 32 代表著我們建立了 32 個過濾器來篩選特徵)
                                       # [filter_height , filter_width , in_channels, output_channels]
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size [28x28x32]
h_pool1 = max_pool_2x2(h_conv1)                          # output size 14x14x32



## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                          # output size 7x7x64



## fc1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  #所以-1，就是缺省值，就是先以你們合適，到時總數除以你們幾個的乘積，我該是幾就是幾
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # output size [1,1024]


## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # output size [1,10]



cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)


session = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.VERSION).split('.')[1]) < 12 and int((tf.VERSION).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
session.run(init)

for i in range(500):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
        



# Store variable
_W_conv1 = W_conv1.eval(session)
_b_conv1 = b_conv1.eval(session)        
_W_conv2 = W_conv2.eval(session)
_b_conv2 = b_conv2.eval(session)     
_W_fc1 = W_fc1.eval(session)
_b_fc1 = b_fc1.eval(session)     
_W_fc2 = W_fc2.eval(session)
_b_fc2 = b_fc2.eval(session)     

        
session.close()




#export model

g_2 = tf.Graph()
with g_2.as_default():
    x_export = tf.placeholder(tf.float32, [None, 28*28], name="input")
    x_image_export = tf.reshape(x_export, [-1, 28, 28, 1])
    
    #conv1    
    W_conv1_export = tf.constant( _W_conv1, name="constant_W_conv1" )
    b_conv1_export = tf.constant( _b_conv1, name="constant_b_conv1" )
    h_conv1_export = tf.nn.relu(conv2d(x_image_export, W_conv1_export) + b_conv1_export) 
    h_pool1_export = max_pool_2x2(h_conv1_export)   
    
    
    #conv2  
    W_conv2_export = tf.constant( _W_conv2, name="constant_W_conv2" )
    b_conv2_export = tf.constant( _b_conv2, name="constant_b_conv2" )
    h_conv2_export = tf.nn.relu(conv2d(h_pool1_export, W_conv2_export) + b_conv2_export) 
    h_pool2_export = max_pool_2x2(h_conv2_export)                
    
    
    ## fc1 layer ##
    W_fc1_export = tf.constant( _W_fc1, name="constant_W_fc1" )
    b_fc1_export = tf.constant( _b_fc1, name="constant_b_fc1" )
    
    h_pool2_flat_export = tf.reshape(h_pool2_export, [-1, 7*7*64])  
    h_fc1_export = tf.nn.relu(tf.matmul(h_pool2_flat_export, W_fc1_export) + b_fc1_export)
                         
    
    ## fc2 layer ##
    W_fc2_export = tf.constant( _W_fc2, name="constant_W_fc2" )
    b_fc2_export =tf.constant( _b_fc2, name="constant_b_fc2" )
    prediction = tf.nn.softmax(tf.matmul(h_fc1_export, W_fc2_export) + b_fc2_export,name="output") # output size [1,10]
    


    sess_2 = tf.Session()
    sess_2.run(tf.global_variables_initializer())
    
    
    graph_def = g_2.as_graph_def()
     
    tf.train.write_graph(graph_def, './model/beginner-export',
                                     'beginner-graph-CNN.pb', as_text=False)
    sess_2.close()

    print("export data")

