# -*- coding:utf-8 -*-
import logging
import time
import tensorflow as tf
from build_rnn import AFD_RNN
from utils import parser_cfg_file
from data_load import DataLoad

class AFD_RNN_Train(object):

    def __init__(self, train_config):

        self.learing_rate = float(train_config['learning_rate'])
        self.train_iterior = int(train_config['train_iteration'])
        self._train_logger_init()

        net_config = parser_cfg_file('./config/rnn_net.cfg')
        self.rnn_net = AFD_RNN(net_config)
        self.predict = self.rnn_net.build_net_graph()
        self.label = tf.placeholder(tf.float32, [None, self.rnn_net.time_step, self.rnn_net.class_num])

    def _compute_loss(self):
        with tf.name_scope('loss'):
            # [batchszie, time_step, class_num] ==> [time_step][batchsize, class_num]
            predict = tf.unstack(self.predict, axis=0)
            label = tf.unstack(self.label, axis=1)

            loss = [tf.nn.softmax_cross_entropy_with_logits(labels=label[i], logits=predict[i]) for i in range(self.rnn_net.time_step) ]
            loss = tf.reduce_mean(loss)
            train_op = tf.train.AdamOptimizer(self.learing_rate).minimize(loss)
        return loss, train_op

    def train_rnn(self):

        loss, train_op = self._compute_loss()

        with tf.name_scope('accuracy'):
            predict = tf.transpose(self.predict, [1,0,2])
            correct_pred = tf.equal(tf.argmax(self.label, 2), tf.argmax(predict, axis=2))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        dataset = DataLoad('./dataset/train/', time_step=self.rnn_net.time_step, class_num= self.rnn_net.class_num)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(1, self.train_iterior+1):
                x, y = dataset.get_batch(self.rnn_net.batch_size)
                if step == 1:
                    feed_dict = {self.rnn_net.input_tensor: x, self.label: y}
                else:
                    feed_dict = {self.rnn_net.input_tensor: x, self.label: y, self.rnn_net.cell_state:state}
                _, compute_loss, state = sess.run([train_op, loss, self.rnn_net.cell_state], feed_dict=feed_dict)

                if step%10 == 0:
                    compute_accuracy = sess.run(accuracy, feed_dict=feed_dict)
                    self.train_logger.info('train step = %d,loss = %f,accuracy = %f'%(step, compute_loss, compute_accuracy))
                if step%1000 == 0:
                    save_path = saver.save(sess, './model/model.ckpt')
                    self.train_logger.info("train step = %d ,model save to =%s" % (step, save_path))

    def _train_logger_init(self):
        """
        初始化log日志
        :return:
        """
        self.train_logger = logging.getLogger('train')
        self.train_logger.setLevel(logging.DEBUG)

        # 添加文件输出
        log_file = './train_logs/' + time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.logs'
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler.setFormatter(file_formatter)
        self.train_logger.addHandler(file_handler)

        # 添加控制台输出
        consol_handler = logging.StreamHandler()
        consol_handler.setLevel(logging.DEBUG)
        consol_formatter = logging.Formatter('%(message)s')
        consol_handler.setFormatter(consol_formatter)
        self.train_logger.addHandler(consol_handler)

if __name__ == '__main__':
    train_config = parser_cfg_file('./config/train.cfg')
    train = AFD_RNN_Train(train_config)
    train.train_rnn()

    # a = tf.zeros([1,2,3])
    # b = tf.unstack(a, axis=1)
    # c = tf.zeros([2,1,3])
    # sess = tf.Session()
    # d = b[0]
    # print(sess.run(b[0]))
    #
    # pass