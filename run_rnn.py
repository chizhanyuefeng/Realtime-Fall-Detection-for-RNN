# -*- coding:utf-8 -*-
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from build_rnn import AFD_RNN
from data_load import DataLoad
from utils import parser_cfg_file

Label = {1: 'Standing', 2: 'Walking', 3: 'Joging', 4: 'Jumping', 5: 'Up stair', 6: 'Down stair', 7: 'stand to sit',
         8: 'Siting', 9: 'Sit to stand', 10: 'Lying', 0: 'Falling', 15: 'CSI', 16: 'CSO'}

class Run_AFD_RNN(object):

    def __init__(self, mode_dir, time_step=5, batch_size=1):
        self.batch_size = batch_size
        self.time_step = time_step

        ckpt = tf.train.get_checkpoint_state(mode_dir)
        if ckpt is None:
            raise FileExistsError(str(mode_dir, '没有模型可以加载'))

        net_config = parser_cfg_file('./config/rnn_net.cfg')
        self.rnn_net = AFD_RNN(net_config, batch_size, time_step)
        predict = self.rnn_net.build_net_graph()
        self._predict_tensor = tf.argmax(predict, axis=2)
        saver = tf.train.Saver()
        self._sess = tf.Session()
        # 加载参数
        saver.restore(self._sess, ckpt.model_checkpoint_path)

    def run(self, data):
        data = np.reshape(data, [self.batch_size, self.time_step, self.rnn_net.senor_data_num])
        predict = self._sess.run(self._predict_tensor, feed_dict={self.rnn_net.input_tensor: data})
        return predict

    def run_stop(self):
        self._sess.close()

    def _update_show_data(self, data, step, update_data):
        for i in range(step):
            data.pop(0)
            data.append(update_data[i])

    def draw_flow(self, test_data, test_label):
        data_size = test_data.shape[0]

        x = [_ for _ in range(150)]
        ax = [0 for _ in range(150)]
        ay = [0 for _ in range(150)]
        az = [0 for _ in range(150)]

        run_step = 10
        num = int(data_size / run_step)

        start_time = time.time()

        plt.axis([0, 151, -20, 20])
        plt.ion()
        for i in range(num):
            if i > int(time_step/run_step):
                predict = run.run(test_data[i * run_step - time_step: i * run_step, :])
                title = 'correct:' + Label[test_label[i * run_step]] + '     predict:' + Label[predict[int(time_step - 1)][0]]
            else:
                title = 'correct:' + Label[test_label[i * run_step]] + '     predict:' + 'unknow'

            self._update_show_data(ax, run_step, test_data[i * run_step:i * run_step + run_step, 0])
            self._update_show_data(ay, run_step, test_data[i * run_step:i * run_step + run_step, 1])
            self._update_show_data(az, run_step, test_data[i * run_step:i * run_step + run_step, 2])

            plt.cla()
            plt.plot(x, ax)
            plt.plot(x, ay)
            plt.plot(x, az)

            plt.title(title)
            plt.draw()
            plt.pause(0.001)

        during = str(time.time() - start_time)
        print('检测耗时=', during)

if __name__ == '__main__':
    net_config = parser_cfg_file('./config/rnn_net.cfg')
    time_step = 50
    class_num = int(net_config['class_num'])

    run = Run_AFD_RNN('./model/', time_step=time_step)
    data_load = DataLoad('./dataset/test/', time_step=time_step, class_num=class_num)

    test_data, test_label = data_load.get_test_data()
    run.draw_flow(test_data, test_label)

    run.run_stop()
