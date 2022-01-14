# MLP_XOR.py
import tensorflow as tf
import numpy as np

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

X = tf.placeholder(tf.float32, [None, 2]) # None은 갯수. 얼마나 들어올지 모름
Y = tf.placeholder(tf.float32, [None, 2]) # placeholder = 담는 그릇

W1 = tf.Variable(tf.random_uniform([2, 3], -1., 1.))
b1 = tf.Variable(tf.random_uniform([3], -1., 1.))
L1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_uniform([3, 2], -1., 1.))
b2 = tf.Variable(tf.random_uniform([2], -1., 1.))
logits = tf.matmul(L1, W2) + b2

output = tf.nn.sigmoid(logits)

cost = -tf.reduce_mean(Y*tf.log(output) + (1 - Y)*tf.log(1 - output))
opt = tf.train.GradientDescentOptimizer(learning_rate = 1)
train_op = opt.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(8000):
        _, cost_val = sess.run([train_op, cost], feed_dict={X:x, Y:y})
        print(step, cost_val)
    print("predcit :", sess.run(logits, feed_dict={X:x}))