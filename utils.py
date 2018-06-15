# -*- coding:utf-8 -*-
import os
import pandas as pd
import numpy as np
import configparser as cp

RAW_DATA_PATH = '/home/tony/fall_research/fall_data/MobiAct_Dataset_v2.0/Annotated Data/'

Label = {'STD':1, 'WAL':2, 'JOG':3, 'JUM':4, 'STU':5, 'STN':6, 'SCH':7, 'SIT':8, 'CHU':9,
         'CSI':10, 'CSO':11, 'LYI':12, 'FOL':13, 'FKL':14, 'BSC':15, 'SDL':16}

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

    save_path = '../dataset/raw/' + os.path.abspath(os.path.dirname(data_file)+os.path.sep+".").replace(RAW_DATA_PATH, '')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = '../dataset/raw/' + data_file.replace(RAW_DATA_PATH, '')
    extract_data.to_csv(save_path, index=0)

def find_all_csv(path):
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
            find_all_csv(path+"/"+i)


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


def main():
    find_all_csv(RAW_DATA_PATH)

if __name__ == '__main__':
    main()