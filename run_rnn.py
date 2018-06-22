# -*- coding:utf-8 -*-
import tensorflow as tf
from build_rnn import AFD_RNN
from data_load import DataLoad
from utils import parser_cfg_file

def run_net(mode_dir, data):

    ckpt = tf.train.get_checkpoint_state(mode_dir)
    if ckpt is None:
        raise FileExistsError(str(mode_dir, '没有模型可以加载'))

    batch_size = data.shape[0]
    net_config = parser_cfg_file('./config/rnn.net.cfg')
    rnn_net = AFD_RNN(net_config, batch_size)
    predict_tensor = rnn_net.build_net_graph()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 加载参数
        saver.restore(sess, ckpt.model_checkpoint_path)

        predict = sess.run(predict_tensor, feed_dict={rnn_net.input_tensor: data})

        print(predict)

if __name__ == '__main__':
    net_config = parser_cfg_file('./config/rnn.net.cfg')
    time_step = int(net_config['time_step'])
    class_num = int(net_config['class_num'])

    data_load = DataLoad('./dataset/test/', time_step, class_num)
    x, y = data_load.get_batch(1)
    run_net('./model/', x)