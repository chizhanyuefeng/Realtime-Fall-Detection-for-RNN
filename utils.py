# -*- coding:utf-8 -*-
import os
import cv2
import pandas as pd
import numpy as np
import configparser as cp
import matplotlib.pyplot as plt

RAW_DATA_PATH = '/home/tony/fall_research/fall_data/MobiAct_Dataset_v2.0/Annotated Data/'

Label = {'STD': 1, 'WAL': 2, 'JOG': 3, 'JUM': 4, 'STU': 5, 'STN': 6, 'SCH': 7, 'SIT': 8, 'CHU': 9,
         'LYI': 10, 'FOL': 0, 'FKL': 0, 'BSC': 0, 'SDL': 0, 'CSI': 15, 'CSO': 16}

def extract_data(data_file, sampling_frequency):
    """
    从mobileFall中提取数据，用于做实验测试
    :param data_file:  原始数据文件
    :param sampling_frequency: 原始数据采集频率
    :return:
    """
    data = pd.read_csv(data_file, index_col=0)
    data_size = len(data.label)
    for i in range(data_size):
        data.iat[i, 10] = Label[data.iloc[i, 10]]

    col_data = np.arange(0, data_size, int(sampling_frequency/50))
    extract_data = data.iloc[col_data, [1, 2, 3, 4, 5, 6, 10]]

    save_path = './dataset/raw/' + os.path.abspath(os.path.dirname(data_file)+os.path.sep+".").replace(RAW_DATA_PATH, '')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = './dataset/raw/' + data_file.replace(RAW_DATA_PATH, '')
    extract_data.to_csv(save_path, index=0)

def find_all_data_and_extract(path):
    """
    递归的查找所有文件并进行转化
    :param path:
    :return:
    """
    if not os.path.exists(path):
        print('路径存在问题：', path)
        return None

    for i in os.listdir(path):
        if os.path.isfile(path+"/"+i):
            if 'csv' in i:
                extract_data(path+"/"+i, 200)
        else:
            find_all_data_and_extract(path+"/"+i)

def parser_cfg_file(cfg_file):
    """
    读取配置文件中的信息
    :param cfg_file: 文件路径
    :return:
    """
    content_params = {}

    config = cp.ConfigParser()
    config.read(cfg_file)

    for section in config.sections():
        # 获取配置文件中的net信息
        if section == 'net':
            for option in config.options(section):
                content_params[option] = config.get(section,option)

        # 获取配置文件中的train信息
        if section == 'train':
            for option in config.options(section):
                content_params[option] = config.get(section,option)

    return content_params

def show_data(data, name=None):
    '''
    show data
    :param data: DataFrame
    :return:
    '''
    num = data.acc_x.size

    x = np.arange(num)
    fig = plt.figure(1, figsize=(100, 60))
    # 子表1绘制加速度传感器数据
    plt.subplot(2, 1, 1)
    plt.title('acc')
    plt.plot(x, data.acc_x, label='x')
    plt.plot(x, data.acc_y, label='y')
    plt.plot(x, data.acc_z, label='z')

    # 添加解释图标
    plt.legend()
    x_flag = np.arange(0, num, num / 10)
    plt.xticks(x_flag)

    # 子表2绘制陀螺仪传感器数据
    plt.subplot(2, 1, 2)
    plt.title('gyro')
    plt.plot(x, data.gyro_x, label='x')
    plt.plot(x, data.gyro_y, label='y')
    plt.plot(x, data.gyro_z, label='z')

    plt.legend()
    plt.xticks(x_flag)
    #plt.show()
    if name is None:
        plt.show()
    else:
        plt.savefig(name)
    plt.close()

def kalman_filter(data):
    kalman = cv2.KalmanFilter(6, 6)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0],
                                         [0, 0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 0, 1]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0],
                                         [0, 0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0],
                                       [0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 1]], np.float32) * 0.003
    kalman.measurementNoiseCov = np.array([[1, 0, 0, 0, 0, 0],
                                          [0, 1, 0, 0, 0, 0],
                                          [0, 0, 1, 0, 0, 0],
                                          [0, 0, 0, 1, 0, 0],
                                          [0, 0, 0, 0, 1, 0],
                                          [0, 0, 0, 0, 0, 1]], np.float32) * 1

    row_num = data.acc_x.size

    for i in range(row_num):
        correct = np.array(data.iloc[i, 0:6].values, np.float32).reshape([6, 1])
        kalman.correct(correct)
        predict = kalman.predict()
        data.iloc[i, 0] = predict[0]
        data.iloc[i, 1] = predict[1]
        data.iloc[i, 2] = predict[2]
        data.iloc[i, 3] = predict[3]
        data.iloc[i, 4] = predict[4]
        data.iloc[i, 5] = predict[5]

    return data

def find_all_data_and_filtrate(path):
    """
    递归的查找所有文件并进行kalman过滤
    :param path:
    :return:
    """
    if not os.path.exists(path):
        print('路径存在问题：', path)
        return None

    for i in os.listdir(path):
        if os.path.isfile(path+"/"+i):
            if 'csv' in i:
                data = pd.read_csv(path+"/"+i)
                data = kalman_filter(data)
                data.to_csv(path+"/"+i, index=False)
        else:
            find_all_data_and_filtrate(path+"/"+i)

def main():
    #find_all_data_and_extract(RAW_DATA_PATH)
    find_all_data_and_filtrate('./dataset/kalman/')

if __name__ == '__main__':
    main()
    # if os.path.exists('./dataset/train/BSC_1_1_annotated.csv') == False:
    #     print('./dataset/train/BSC_1_1_annotated.csv', '文件不存在！')
    # data = pd.read_csv('./dataset/train/BSC_1_1_annotated.csv')
    #
    # #show_data(data)
    # data = kalman_filter(data)
    # data.to_csv('./dataset/train/BSC_1_1_annotated.csv', index=False)
    # #show_data(data)
    # # a = data.iloc[4:5,0]
    # # print(a)
    # data = pd.read_csv('./dataset/train/STU_1_1_annotated.csv')
    #
    # show_data(data)

