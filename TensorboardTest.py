import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):
    
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            #觀察想觀察的變量
            tf.summary.histogram(layer_name+'/Weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1, name='b')
            tf.summary.histogram(layer_name+'/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights)+biases
    
        
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b) 
        tf.summary.histogram(layer_name+'/outputs', outputs)   
        return outputs


x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)#為了讓數據更像真實數據
y_data = np.square(x_data)-0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1], name='x_input')
    ys = tf.placeholder(tf.float32,[None,1], name='y_input')

#add hidden layer
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function = tf.nn.relu)
#add output layer
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), 1))
    tf.summary.scalar('loss', loss)

optimizer = tf.train.GradientDescentOptimizer(0.1)
with tf.name_scope('train_step'):
    train_step = optimizer.minimize(loss)
init = tf.global_variables_initializer()
  
session = tf.Session()


# 將視覺化輸出
merged = tf.summary.merge_all()
#writer = tf.train.SummaryWriter("logs/",session.graph) # is deprecated, instead use tf.summary.FileWriter
writer = tf.summary.FileWriter("logs/",session.graph)

session.run(init)

for i in range(1000):
    session.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i%50 == 0:
        result = session.run(merged, feed_dict={xs:x_data, ys:y_data})
        writer.add_summary(result, i)


writer.close()
session.close()


#1.session run起來後，初始化
#   merged = tf.summary.merge_all()
#   writer = tf.summary.FileWriter("logs/",session.graph)
#2.建立graphs
#   a)給node name
#   b)然後可用with tf.name_scope('train_step'): 包起來
#3.建立scalars
#   writer.add_summary(result, i) 填入值和step，可以看loss的變化，或是你想看的其他數值的變化


#tensorflow 1.0.0
#tensorboard --logdir=logs --debug

#tensorflow 1.2.0
#writer = tf.summary.FileWriter("logs/"+NAME,session.graph)
#tensorboard --logdir=NAME:PATH
#範例:tensorboard --logdir=:D:\workspace_python\TensorflowTest\logs
#如果需要跑多個tensorboard --logdir=test:D:\workspace_python\TensorflowTest\logs,train:D:\workspace_python\TensorflowTest\logs