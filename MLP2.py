# MLP2.py, XOR proglem solution with Softmax practice
import tensorflow as tf
import numpy as np

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[1, 0], [0, 1], [0, 1], [1, 0]] # [and, xor], one-hot vector

X = tf.placeholder(tf.float32, [None, 2]) # None은 갯수. 얼마나 들어올지 모름
Y = tf.placeholder(tf.float32, [None, 2]) # placeholder = 담는 그릇

W1 = tf.Variable(tf.random_uniform([2, 3], -1., 1.))
b1 = tf.Variable(tf.random_uniform([3], -1., 1.))
L1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_uniform([3, 2], -1., 1.))
b2 = tf.Variable(tf.random_uniform([2], -1., 1.))
logits = tf.matmul(L1, W2) + b2

output_softmax = tf.nn.softmax(logits) # 각각의 i번째 x가 y_i에 맞을 확률들
output_argmax = tf.argmax(logits, 1)
# tf.argmax 두번째 인자값의 범위는 [-rank(input), rank(input))로 한정.
# 텐서플로우에서 rank는 텐서의 원소 하나에 접근하기 위해 필요한 인덱스의 개수

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
opt = tf.train.GradientDescentOptimizer(learning_rate = 1)
train_op = opt.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(8000):
        _, cost_val = sess.run([train_op, cost], feed_dict={X:x, Y:y})
        if step % 100 == 0:
            print(step, cost_val)
    print("predcit :", sess.run(logits, feed_dict={X:x}))
    print("predcit with softmax :", sess.run(output_softmax, feed_dict={X:x}))
    print("predcit with argmax :", sess.run(output_argmax, feed_dict={X:x}))