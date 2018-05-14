import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function = None):
    
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs, Weights)+biases
#     print("inputs = ",inputs)
#     print("Weights = ",Weights)
#     print("biases = ",biases)
#     print("Wx_plus_b = ",Wx_plus_b)
    
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)    
    return outputs


x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)#為了讓數據更像真實數據
y_data = np.square(x_data)-0.5 + noise


xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

#add hidden layer
l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
#add output layer
prediction = add_layer(l1,10,1,activation_function=None)
 
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), 1))
#square : [1, 2, 3] = [1, 4, 9]
#reduce_sum : [1, 2, 3] = 6
#reduce_mean : [1, 2, 3] = 2

optimizer = tf.train.GradientDescentOptimizer(0.1)
train_step = optimizer.minimize(loss)
  
session = tf.Session()
session.run(tf.global_variables_initializer())

# matplotlib start
fig  = plt.figure() #生成圖片框
ax = fig.add_subplot(1,1,1) #連續性的畫圖(編號)
ax.scatter(x_data,y_data)
# plt.ion()# 為了讓plot可以繼續畫

# matplotlib end

for i in range(1000):
    session.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i%50 == 0:
        print(session.run(loss,feed_dict={xs:x_data, ys:y_data}))

        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = session.run(prediction ,feed_dict={xs:x_data})
        lines = ax.plot(x_data, prediction_value,'r-',lw=5)
        plt.pause(0.1)

plt.show()
session.close()
#tf.square([1 2 3])=[1 4 9]
#tf.reduce_sum
