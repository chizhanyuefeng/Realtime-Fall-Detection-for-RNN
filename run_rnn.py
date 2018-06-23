# -*- coding:utf-8 -*-
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from build_rnn import AFD_RNN
from data_load import DataLoad
from utils import parser_cfg_file



Label = {1: 'STD', 2: 'WAL', 3: 'JOG', 4: 'JUM', 5: 'STU', 6: 'STN', 7: 'SCH', 8: 'SIT', 9: 'CHU',
         10: 'LYI', 0: 'FALL', 15: 'CSI', 16: 'CSO'}

class Run_AFD_RNN(object):

    def __init__(self, mode_dir, batch_size=1, time_step=5):
        ckpt = tf.train.get_checkpoint_state(mode_dir)
        if ckpt is None:
            raise FileExistsError(str(mode_dir, '没有模型可以加载'))

        #batch_size = data.shape[0]
        net_config = parser_cfg_file('./config/rnn_net.cfg')
        self.rnn_net = AFD_RNN(net_config, batch_size, time_step)
        predict = self.rnn_net.build_net_graph()
        self._predict_tensor = tf.argmax(predict, axis=2)
        saver = tf.train.Saver()
        self._sess = tf.Session()
        # 加载参数
        saver.restore(self._sess, ckpt.model_checkpoint_path)

    def run(self, data):
        predict = self._sess.run(self._predict_tensor, feed_dict={self.rnn_net.input_tensor: data})
        return predict

    def run_stop(self):
        self._sess.close()

def update_show_data(data, step, update_data):
    for i in range(step):
        data.pop(0)
        data.append(update_data[i])

if __name__ == '__main__':
    net_config = parser_cfg_file('./config/rnn_net.cfg')
    time_step = int(net_config['time_step'])
    class_num = int(net_config['class_num'])

    data_load = DataLoad('./dataset/test/', 2, class_num)
    x, y = data_load.get_batch(1, [2])
    # print(y.argmax(2))
    # run_net('./model/', x)
    print(x)
    run = Run_AFD_RNN('./model/', time_step=2)
    predict = run.run(x)
    print(predict)

    # test_data, test_label = data_load.get_test_data()
    # data_size = test_data.shape[0]
    # plt.axis([0, 151, -20, 20])
    # plt.ion()
    #
    #
    # x = [_ for _ in range(150)]
    # ax = [0 for _ in range(150)]
    # ay = [0 for _ in range(150)]
    # az = [0 for _ in range(150)]
    #
    # num = int(data_size / 5)
    # print(data_size)
    #
    # start_time = time.time()
    #
    # for i in range(num):
    #
    #     update_show_data(ax, 5, test_data[i*5:i*5+5,0])
    #     update_show_data(ay, 5, test_data[i * 5:i * 5 + 5, 1])
    #     update_show_data(az, 5, test_data[i * 5:i * 5 + 5, 2])
    #
    #     plt.cla()
    #     plt.plot(x, ax)
    #     plt.plot(x, ay)
    #     plt.plot(x, az)
    #     plt.title(Label[test_label[i*5]])
    #     plt.draw()
    #     plt.pause(0.000001)
    # during = str(time.time() - start_time)
    # print('检测耗时=', during)