import tensorflow as tf



state = tf.Variable(0, name = 'counter')


one = tf.constant(1)
# new_value = tf.add(state,one)
new_value = state+one      #tf.add(x, y) and x + y are equivalent. 只是有function name 較好讀
update = tf.assign(state, new_value)


# init = tf.initialize_all_variables() #2017-03-02過期
init = tf.global_variables_initializer() 

with tf.Session() as session:
    session.run(init)
    for _ in range(3):
        print(  session.run(update))