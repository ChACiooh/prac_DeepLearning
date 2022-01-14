# MLP1.py
import tensorflow as tf

x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_uniform([3, 2], -1.0, 1.0))
b1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

Z1 = tf.add(tf.matmul(W1, tf.transpose(X)), b1)
A1 = tf.nn.relu(Z1)

W2 = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))
b2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

Z2 = tf.add(tf.matmul(W2, A1), b2)
A2 = tf.nn.sigmoid(Z2)

cost = tf.reduce_mean(tf.square(Y - A2))
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = opt.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(100):
        for x, y in zip(x_data, y_data):
            _, cost_val = sess.run([train_op, cost], feed_dict={X: [x], Y: [y]})
        print("step:{} cost:{}".format(step+1, cost_val))
        print("W1:{} b1:{}".format(sess.run(W1), sess.run(b1)))
        print("W2:{} b2:{}".format(sess.run(W2), sess.run(b2)))
        print('')
            
    print(sess.run(A2, feed_dict={X:x_data}))