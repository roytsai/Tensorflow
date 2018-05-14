'''
Created on 2018年3月26日

@author: RoyKT_Tsai
'''

import tensorflow as tf
import datetime, time
from nt import write
from tensorflow.examples.tutorials.mnist import input_data
from absl.logging import log

Weights = 0
biases = 0

def setup_traing_data(source, des_x, dex_y):
    new_data = []
    batch_xs = []
    batch_ys = []
    with open(source, 'r', encoding ='utf-8') as output:
        data_list = output.read().split('\n')
        data_list.pop(0)
        with open(dex_y, 'w', encoding ='utf-8') as writer:
            size = len(data_list)
            
            for index, line in enumerate(data_list):
                elements = line.split(',')
                new_line = [int(x) for x in elements]
                new_data.append(new_line)
                y = [new_line[4],new_line[5],new_line[6],new_line[7],new_line[8],new_line[9]]  
                if index ==  size-1 :
                    writer.write(str(y))
                else:
                    writer.write(str(y)+'\n')
                batch_ys.append(y)    
        writer.close();        
    output.close()   
    
    with open(des_x, 'w', encoding ='utf-8') as writer:
        size = len(new_data)
        for index,line in enumerate(new_data):
            
            hour_binary = '{0:05b}'.format(line[3]) 
            hour_binary_list = [int(x) for x in hour_binary]
            raw_data = []
            raw_data.extend(hour_binary_list)
            for i in range(index-1, index-11, -1): #自己不算
                
                data = []
                if i >= 0:
                    data = [new_data[i][4],new_data[i][5],new_data[i][6],new_data[i][7],new_data[i][8],new_data[i][9]]
                else :
                    data =[0,0,0,0,0,0]
                    
                raw_data.extend(data)

            if index ==  size-1 :
                writer.write(str(raw_data))    
            else:
                writer.write(str(raw_data)+'\n')   
            batch_xs.append(raw_data)
    writer.close()
    return batch_xs, batch_ys
 
def add_layer(inputs, in_size, out_size, activation_function = None):
    global Weights
    global biases
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name="Variable_W")
    biases = tf.Variable(tf.zeros([1,out_size])+0.1, name="Variable_b") 
    Wx_plus_b = tf.matmul(inputs, Weights)+biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, 0.6)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)    
    return outputs
       
     
#define placeholder for input to n
xs = tf.placeholder(tf.float32, [None,65])
ys = tf.placeholder(tf.float32, [None,6])
# xs = tf.placeholder(tf.float32, [None,784])
# ys = tf.placeholder(tf.float32, [None,10])
 
 
Weights = tf.Variable([0.0],tf.float32)
biases =tf.Variable([0.0],tf.float32)  
#add output layer
prediction = add_layer(xs, 65, 6, activation_function = tf.nn.softmax)
# prediction = add_layer(xs, 784, 10, activation_function = tf.nn.softmax)
#softmax 這個激勵函數通常是用來做classifcation 
 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1])) 
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)  #MomentumOptimizer :考慮上一步的learning rate, 


session = tf.Session()
session.run(tf.global_variables_initializer())
print('global_variables_initializer')




if __name__ == '__main__':

    training_batch_xs, training_batch_ys = setup_traing_data('training_data.txt','training_data_x.txt','training_data_y.txt')
    testing_batch_xs, testing_batch_ys = setup_traing_data('test_data.txt','test_data_x.txt','test_data_y.txt')


    for i in range(5000):
        session.run(train_step, feed_dict={xs:training_batch_xs, ys:training_batch_ys})
        if i%200 == 0:
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))  
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(session.run(accuracy, feed_dict={xs: testing_batch_xs, ys: testing_batch_ys}))
    




    # Store variable
    _W = Weights.eval(session)
    _b = biases.eval(session)
    g_2 = tf.Graph()
    with g_2.as_default():
        x_2 = tf.placeholder(tf.float32, [None, 65], name="input")
        W_2 = tf.constant( _W, name="constant_W")
        b_2 = tf.constant( _b, name="constant_b")
        y_2 = tf.nn.softmax(tf.add(tf.matmul(x_2, W_2),b_2), name="output")

     
        sess_2 = tf.Session()
        sess_2.run(tf.global_variables_initializer())  
        graph_def = g_2.as_graph_def()
          
        tf.train.write_graph(graph_def, './model',
                                         'beginner-graph.pb', as_text=False)
        sess_2.close()

    
    
    