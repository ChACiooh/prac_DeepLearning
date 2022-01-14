import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
import os
from image_show import *


def batch_data(shuffled_idx, batch_size, data, labels, start_idx):
    idx = shuffled_idx[start_idx:start_idx+batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
"""
def build_AE(x):
    x_flat = tf.reshape(x, [-1, 32*32*3])
    aeW1 = tf.get_variable(name="aeW1", shape=[32*32*3, 2 ** 8], initializer=tf.contrib.layers.xavier_initializer())
    aeb1 = tf.get_variable(name="aeb1", shape=[2 ** 8], initializer=tf.contrib.layers.xavier_initializer())
    aeZ1 = tf.add(tf.matmul(x_flat, aeW1), aeb1)
    aeA1 = tf.nn.relu(aeZ1)

    aeW2 = tf.get_variable(name="aeW2", shape=[2 ** 8, 32], initializer=tf.contrib.layers.xavier_initializer())
    aeb2 = tf.get_variable(name="aeb2", shape=[32], initializer=tf.contrib.layers.xavier_initializer())
    aeZ2 = tf.add(tf.matmul(aeA1, aeW2), aeb2)
    encoder = tf.nn.sigmoid(aeZ2)

    adW1 = tf.get_variable(name="adW1", shape=[32, 2 ** 8], initializer=tf.contrib.layers.xavier_initializer())
    adb1 = tf.get_variable(name="adb1", shape=[2 ** 8], initializer=tf.contrib.layers.xavier_initializer())
    adZ1 = tf.add(tf.matmul(encoder, adW1), adb1)
    adA1 = tf.nn.relu(adZ1)

    adW2 = tf.get_variable(name="adW2", shape=[2 ** 8, 32*32*3], initializer=tf.contrib.layers.xavier_initializer())
    adb2 = tf.get_variable(name="adb2", shape=[32*32*3], initializer=tf.contrib.layers.xavier_initializer())
    adZ2 = tf.add(tf.matmul(adA1, adW2), adb2)
    decoder = tf.nn.sigmoid(adZ2)

    return decoder, adZ2
    """

def get_next_size(width, height, fw, fh, sw, sh):
    pw = int((fw - 1) / 2) if fw % 2 == 1 else (fw - 1) / 2
    ph = int((fh - 1) / 2) if fh % 2 == 1 else (fh - 1) / 2

    next_width = width + int(2*pw) - fw
    next_width = next_width - (next_width % sw)
    next_width = int(next_width / sw) + 1

    next_height = height + int(2*ph) - fh
    next_height = next_height - (next_height % sh)
    next_height = int(next_height / sh) + 1
    print('nw:{} nh:{}'.format(next_width, next_height))
    return next_width, next_height

def add_layer(nth, prev_l, prvw, prvh, fw, fh, nin, nout, sw, sh, layer='Conv', padding = 'SAME'):
    nw, nh = get_next_size(prvw, prvh, fw, fh, sw, sh)
    if layer == 'MaxPool':
        logits = tf.nn.max_pool(prev_l, ksize=[1, fh, fw, 1], strides=[1, sh, sw, 1], padding=padding)
        return logits, nw, nh
    else:
        W = tf.get_variable(name="W{}".format(nth), shape=[fh, fw, nin, nout], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="b{}".format(nth), shape=[nout], initializer=tf.contrib.layers.xavier_initializer())
        c = tf.nn.conv2d(prev_l, W, strides=[1, sh, sw, 1], padding=padding)
        logits = tf.nn.relu(tf.nn.bias_add(c, b))
        
        return logits, nw, nh

def add_fc_layer(nth, prev_l, nin, nout):
    W = tf.get_variable(name='W_fc{}'.format(nth), shape=[nin, nout], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name='b_fc{}'.format(nth), shape=[nout], initializer=tf.contrib.layers.xavier_initializer())
    logit = tf.nn.relu(tf.nn.bias_add(tf.matmul(prev_l, W), b))

    return logit

def build_CNN_classifier(x):
    x_image = x
    nw = 32
    nh = 32
    print('x_image :', x_image)

    N = 8

    f = np.zeros(N, dtype=int)
    n = np.zeros(N, dtype=int)
    conv_stride = np.ones(N, dtype=int)
    mpf = np.zeros(N, dtype=int)
    mps = np.ones(N, dtype=int)
    
    f[:] = 3
    mps[-3:] = 2
    for i in range(N):
        n[i] = 64 if i % 2 == 0 else 32
        """if i > 4:
            mps[i] = 2 if i % 2 == 0 else 1"""
    mpf[:] = 3

    logit_list = []
    lpool_list = []
    logit, nw, nh = add_layer(1, x_image, nw, nh, f[0], f[0], 3, n[0], conv_stride[0], conv_stride[0])
    logit_list.append(logit)
    lpool, nw, nh = add_layer(1, logit, nw, nh, mpf[0], mpf[0], 0, 0, mps[0], mps[0], layer='MaxPool')
    lpool_list.append(lpool)
    j = N - 1
    for i in range(1, N):
        logit, nw, nh = add_layer(i+1, lpool, nw, nh, f[i], f[i], n[i-1], n[i], conv_stride[i], conv_stride[i])
        logit_list.append(logit)
        lpool, nw, nh = add_layer(i+1, logit, nw, nh, mpf[i], mpf[i], 0, 0, mps[i], mps[i], layer='MaxPool')
        lpool_list.append(lpool)

    n3 = n[j]
    l3_flat = tf.reshape(lpool_list[j], [-1, nw*nh*n3])
    print(l3_flat)


    ns = [64, 64, 32]
    logits = []
    logit = add_fc_layer(1, l3_flat, nw*nh*n3, ns[0])
    logits.append(logit)
    for i in range(1, len(ns)):
        next_logit = add_fc_layer(i+1, logit, ns[i-1], ns[i])
        logits.append(next_logit)
        logit = next_logit
        
    W_fc = tf.get_variable(name='W_fc', shape=[ns[-1], 10], initializer=tf.contrib.layers.xavier_initializer())
    b_fc = tf.get_variable(name='b_fc', shape=[10], initializer=tf.contrib.layers.xavier_initializer())
    logit = tf.nn.bias_add(tf.matmul(logits[-1], W_fc), b_fc)
    
    hypothesis = tf.nn.softmax(logit)

    return hypothesis, logit

def shift_npx(x, n):
    # 32 * (32, 3)
    x0 = x.reshape(3, 32, 32)
    x1 = np.zeros(x0.shape, dtype=int)
    x1[:, :, n:] = x0[:, :, :-n]
    x1[:, :, :n] = x0[:, :, -n:]

    x2 = np.zeros(x0.shape, dtype=int)
    x2[:, n:, :] = x0[:, :-n, :]
    x2[:, :n, :] = x0[:, -n:, :]

    x3 = np.zeros(x0.shape, dtype=int)
    x3[:, :, :-n] = x0[:, :, n:]
    x3[:, :, -n:] = x0[:, :, :n]

    x4 = np.zeros(x0.shape, dtype=int)
    x4[:, :-n, :] = x0[:, n:, :]
    x4[:, -n:, :] = x0[:, :n, :]

    x1 = x1.reshape((32, 32, 3))
    x2 = x2.reshape((32, 32, 3))
    x3 = x3.reshape((32, 32, 3))
    x4 = x4.reshape((32, 32, 3))
    return x1, x2, x3, x4



# execution part #

# img_show_n(5)     show 5 images for test randomly

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])

x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")

print('x_train :', x_train.shape)
print('y_train :', y_train.shape)

# Augmentation; optional
"""aug_x = []
aug_y = []
for xt, yt in zip(x_train[:3000], y_train[:3000]):
    x1, x2, x3, x4 = shift_npx(xt, 2)
    aug_x.append(x1)
    aug_x.append(x2)
    aug_x.append(x3)
    aug_x.append(x4)
    aug_y.append(yt)
    aug_y.append(yt)
    aug_y.append(yt)
    aug_y.append(yt)

aug_x = np.array(aug_x)
aug_y = np.array(aug_y)

print('aug_x shape :', aug_x.shape)
print('aug_y shape :', aug_y.shape)"""

dev_num = len(x_train) // 4

x_dev = x_train[:dev_num]
y_dev = y_train[:dev_num]

x_train = x_train[dev_num:]
y_train = y_train[dev_num:]

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10),axis=1)
y_dev_one_hot = tf.squeeze(tf.one_hot(y_dev, 10),axis=1)

#x_ = build_AE(x)
y_pred, logits = build_CNN_classifier(x)

training_epochs = 40
batch_size = 32
learning_rate = 0.0003

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

total_batch = int(len(x_train)/batch_size) if len(x_train)%batch_size == 0 else int(len(x_train)/batch_size) + 1

print(train_step)
print(total_batch)

ckpt_path = "output/"

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    print("학습시작")

    if os.path.exists(ckpt_path+'.meta'):
        saver.restore(sess, ckpt_path)
        print("로드 성공")

    else:
        for epoch in range(training_epochs):
            print("Epoch", epoch+1)
            start = 0
            shuffled_idx = np.arange(0, len(x_train))
            np.random.shuffle(shuffled_idx)

            for i in range(total_batch):
                batch = batch_data(shuffled_idx, batch_size, x_train, y_train_one_hot.eval(), i*batch_size)
                sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})
        saver.save(sess, ckpt_path)
    #saver = tf.train.Saver()
    #saver.save(sess, ckpt_path)
    #saver.restore(sess, ckpt_path)

    y_prediction = np.argmax(y_pred.eval(feed_dict={x: x_dev}), 1)
    y_true = np.argmax(y_dev_one_hot.eval(), 1)
    dev_f1 = f1_score(y_true, y_prediction, average="weighted") # f1 스코어 측정
    print("dev 데이터 f1 score: %f" % dev_f1)

    # 밑에는 건드리지 마세요
    x_test = np.load("data/x_test.npy")
    test_logits = y_pred.eval(feed_dict={x: x_test})
    np.save("result", test_logits)
