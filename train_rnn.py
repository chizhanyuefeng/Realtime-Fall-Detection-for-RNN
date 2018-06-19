import tensorflow as tf
from afd_rnn import AFD_RNN
from utils import parser_cfg_file

def get_next_batch():
    x=0
    y=0
    return x,y

def train_rnn():
    train_content = parser_cfg_file('./config/train.cfg')
    learing_rate = float(train_content['learing_rate'])
    train_iterior = int(train_content['train_iterior'])

    rnn_net = AFD_RNN()
    predict = rnn_net.build_net_graph()
    label = tf.placeholder(tf.float32, [None, rnn_net.time_step, rnn_net.class_num])

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=predict))
        train_op = tf.train.AdamOptimizer(learing_rate).minimize(loss)

    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(train_iterior):
            x, y = get_next_batch()
            if step == 0:
                feed_dict = {rnn_net.x:x, label:y}
            else:
                feed_dict = {rnn_net.x: x, label: y, rnn_net.cell_state:state}
            _, loss, state = sess.run([train_op, rnn_net.cell_state], feed_dict=feed_dict)

            if step%100 == 0:
                accuracy = sess.run(accuracy, feed_dict=feed_dict)
                print('train step = %d，loss = %f,accuracy = %f：'%(step, loss, accuracy))
