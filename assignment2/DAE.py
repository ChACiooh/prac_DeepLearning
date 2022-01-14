import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

n_input = 28*28

X = tf.placeholder(tf.float32, [None, n_input])
#noise = tf.Variable(tf.random_uniform([n_input], -1., 1.))
#X_ = tf.add(X, noise)

num_hid_layers = 4
num_hid_dims = np.random.randint(128, 256, size=num_hid_layers)
num_hid_dims = np.insert(num_hid_dims, 0, n_input)

W_enc = []
b_enc = []
W_dec = []
b_dec = []
for i in range(num_hid_layers):
    W_enc.append(tf.Variable(tf.random_uniform([num_hid_dims[i], num_hid_dims[i+1]], -1., 1.)))
    b_enc.append(tf.Variable(tf.random_uniform([num_hid_dims[i+1]], -1., 1.)))
    W_dec.append(tf.Variable(tf.random_uniform([num_hid_dims[num_hid_layers - (i)], 
                                                num_hid_dims[num_hid_layers - (i+1)]], -1., 1.)))
    b_dec.append(tf.Variable(tf.random_uniform([num_hid_dims[num_hid_layers - (i+1)]], -1., 1.)))

enc_h = []
dec_h = []

for i in range(num_hid_layers):
    if i == 0:
        enc_h.append(tf.nn.sigmoid(tf.add(tf.matmul(X, W_enc[i]), b_enc[i]))) # uniform : X대신 X_
    else:
        enc_h.append(tf.nn.sigmoid(tf.add(tf.matmul(enc_h[i-1], W_enc[i]), b_enc[i])))

for i in range(num_hid_layers):
    if i == 0:
        dec_h.append(tf.nn.sigmoid(tf.add(tf.matmul(enc_h[num_hid_layers-1], W_dec[i]), b_dec[i])))
    else:
        dec_h.append(tf.nn.sigmoid(tf.add(tf.matmul(dec_h[i-1], W_dec[i]), b_dec[i])))

learning_rate = 0.01

decoder = dec_h[num_hid_layers-1]
cost = tf.reduce_mean(tf.square(X - decoder))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

batch_size = 100
total_step = int(mnist.train.num_examples/batch_size)
epoch_num = 40
noise_level = 0.6

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epoch_num):
        avg_cost = 0
        for i in range(total_step):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs_noise = batch_xs + noise_level * np.random.normal(loc=0.0, scale=1.0, size=batch_xs.shape)
            _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            avg_cost += cost_val / total_step
        print('Epoch:', '%d' % (epoch+1), 'cost:', '{:.9f}'.format(avg_cost))
        
        if epoch+1 > 10 and (epoch+1)%10 == 0:
            samples = sess.run(decoder, feed_dict={X: mnist.test.images[:10]})
            X_noise = mnist.test.images[:10] + noise_level * np.random.normal(loc=0.0, scale=1.0, size=mnist.test.images[:10].shape)
            fig, ax = plt.subplots(3, 10, figsize=(10, 3))
            #noise_ = np.reshape(noise.eval(), (28, 28))

            for i in range(10):
                ax[0][i].set_axis_off()
                ax[1][i].set_axis_off()
                ax[2][i].set_axis_off()
                ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
                ax[1][i].imshow(np.reshape(X_noise[i], (28, 28))) # uniform : imshow 안에 + noise_
                ax[2][i].imshow(np.reshape(samples[i], (28, 28)))
            plt.show()
