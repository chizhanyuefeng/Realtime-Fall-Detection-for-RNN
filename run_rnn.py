# -*- coding:utf-8 -*-
import tensorflow as tf
from build_rnn import AFD_RNN

def run_net(mode_dir):

    ckpt = tf.train.get_checkpoint_state(mode_dir)
    if ckpt is None:
        print(mode_dir, '没有模型可以加载')
        return None
    # 加载图结构
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

    with tf.Session() as sess:
        # 加载参数
        saver.restore(sess, ckpt.model_checkpoint_path)





if __name__ == '__main__':
    run_net('./model/')