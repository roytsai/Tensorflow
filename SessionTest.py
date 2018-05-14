import tensorflow as tf

matrix1 = tf.constant([[3,3]]) #1*2
matrix2 = tf.constant([[2],
                       [2]]) #2*1

product = tf.matmul(matrix1, matrix2)

#method 1

# session = tf.Session()
# result = session.run(product) # 每run一次才會跑一次上述tensorflow所建構的流程
# print(result)
# session.close()

#method 2
with tf.Session() as session:
    result = session.run(product)
    print(result)