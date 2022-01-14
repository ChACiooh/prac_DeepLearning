import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn_cell
import os
# from text_show import *

def batch_data(shuffled_idx, batch_size, data, labels, start_idx):
    idx = shuffled_idx[start_idx:start_idx+batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def get_vocabulary_size(X):
    return max([max(x) for x in X]) + 1 # plus the 0th word

def fit_in_vocabulary(X, voc_size):
    return [[w for w in x if w < voc_size] for x in X]

def zero_pad(X, seq_len):
    return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X])

def embedding(x, vocab_size, EMBEDDING_DIM):
    embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0, 1.0), trainable=True)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, x)
    return batch_embedded

"""def attention(initial_state, batch_embedded, lstm_outputs, EMBEDDING_DIM):
    encoder = tf.reshape(lstm_outputs, [-1, EMBEDDING_DIM], name='encoder')
    return encoder"""

def build_classifier(batch_embedded, HIDDEN_SIZE, keep_prob):
    # Embedding layer
    print('batch_embedded shape :', batch_embedded)

    # LSTM layer
    cell = rnn_cell.BasicLSTMCell(num_units=HIDDEN_SIZE, state_is_tuple=-True)
    #initial_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, states = tf.nn.dynamic_rnn(cell, batch_embedded, dtype=tf.float32)

    print('rnn_outputs shape :', rnn_outputs)
    print('states shape :', states)

    # Fully connected layer
    x_for_fc = tf.reshape(states, [-1, HIDDEN_SIZE])
    l1 = tf.contrib.layers.fully_connected(x_for_fc, HIDDEN_SIZE, activation_fn=None)
    l1 = tf.nn.dropout(l1, keep_prob)
    l2 = tf.contrib.layers.fully_connected(l1, 2, activation_fn=None)
    logits = tf.reshape(l2, [-1, 2])
    _, logits = tf.split(logits, num_or_size_splits=2, axis=0)
    hypothesis = tf.nn.softmax(logits)
    print('')
    print('hypothesis :', hypothesis)
    print('logits :', logits)
    print('')
    return hypothesis, logits


# Load the data set
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")
x_test = np.load("data/x_test.npy")

np.load = np_load_old


dev_num = len(x_train) // 4
print('dev_num :', dev_num)

x_dev = x_train[:dev_num]
y_dev = y_train[:dev_num]
print(x_train[0])
print(y_train[0])

x_train = x_train[dev_num:]
y_train = y_train[dev_num:]

x_len = len(x_train)
max_len_x = max([len(i) for i in x_train])
avg_len_x = 0
for i in x_train:
    avg_len_x += len(i) / x_len
print('max len x', max_len_x)
print('avg len x', avg_len_x)

# Hyperparameters

ckpt_path = "output/"

SEQUENCE_LENGTH = int(avg_len_x) + 1
EMBEDDING_DIM = int(avg_len_x) + 1
HIDDEN_SIZE = 256
BATCH_SIZE = 32
NUM_EPOCHS = 10
learning_rate = 0.001

print('SEQ LEN :', SEQUENCE_LENGTH)
print('EMB DIM :', EMBEDDING_DIM)


# Sequences pre-processing
vocabulary_size = get_vocabulary_size(x_train)
x_dev = fit_in_vocabulary(x_dev, vocabulary_size)
x_train = zero_pad(x_train, SEQUENCE_LENGTH)
x_dev = zero_pad(x_dev, SEQUENCE_LENGTH)

batch_ph = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='batch_ph')
target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
batch_embedded = embedding(batch_ph, vocabulary_size, EMBEDDING_DIM)

print('vocab size', vocabulary_size)
print('batch_ph', batch_ph)
print('target_ph', target_ph)


y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 2))
y_dev_one_hot = tf.squeeze(tf.one_hot(y_dev, 2))

y_pred, logits = build_classifier(batch_embedded, HIDDEN_SIZE, keep_prob=keep_prob)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_ph, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Accuracy metric
is_correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(target_ph, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


total_batch = int(len(x_train)/BATCH_SIZE) if len(x_train)%BATCH_SIZE == 0 else int(len(x_train)/BATCH_SIZE) + 1

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    print("학습시작")

    if os.path.exists(ckpt_path+'.meta'):
        saver.restore(sess, ckpt_path)
        print("로드 성공")

    else:
        for epoch in range(NUM_EPOCHS):
            print("Epoch", epoch + 1)
            start = 0
            shuffled_idx = np.arange(0, len(x_train))
            np.random.shuffle(shuffled_idx)

            for i in range(total_batch):
                batch = batch_data(shuffled_idx, BATCH_SIZE, x_train, y_train_one_hot.eval(), i * BATCH_SIZE)
                #print('batch[0] :', batch[0])
                #print('batch[0] shape :', batch[0].shape)
                sess.run(optimizer, feed_dict={batch_ph: batch[0], target_ph: batch[1], keep_prob:0.5})
            saver.save(sess, ckpt_path)
            #saver.restore(sess, ckpt_path)

    dev_accuracy = accuracy.eval(feed_dict={batch_ph: x_dev, target_ph: np.asarray(y_dev_one_hot.eval()), keep_prob:1})
    print("dev 데이터 Accuracy: %f" % dev_accuracy)

    # 밑에는 건드리지 마세요
    x_test = fit_in_vocabulary(x_test, vocabulary_size)
    x_test = zero_pad(x_test, SEQUENCE_LENGTH)

    test_logits = y_pred.eval(feed_dict={batch_ph: x_test, keep_prob:1.0})
    np.save("result", test_logits)