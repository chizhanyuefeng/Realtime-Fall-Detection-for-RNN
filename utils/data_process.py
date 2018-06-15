# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

RAW_DATA_PATH = '/home/tony/fall_research/fall_data/MobiAct_Dataset_v2.0/Annotated Data/'
Label = {'STD':1,'WAL':2,'JOG':3,'JUM':4,'STU':5,'STN':6,'SCH':7,'SIT':8,'CHU':9,'CSI':10,
         'CSO':11,'LYI':12,'FOL':0,'FKL':0,'BSC':0,'SDL':0}

def extract_data(data_file, sampling_frequency):
    """
    从mobileFall中提取数据，用于做实验测试
    :param data_file:  原始数据文件
    :param sampling_frequency: 原始数据采集频率
    :return:
    """
    data = pd.read_csv(data_file, index_col=0)

    col_data = np.arange(0, len(data.label), int(sampling_frequency/50))
    extract_data = data.iloc[col_data, [1, 2, 3, 4, 5, 6, 10]]
    extract_data.to_csv('../dataset/test.csv', index=0)


def main():
    extract_data(RAW_DATA_PATH + 'STU/STU_1_1_annotated.csv', 200)

if __name__ == '__main__':
    main()