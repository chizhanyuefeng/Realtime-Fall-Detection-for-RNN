import os
import random
import numpy as np
import pandas as pd

class DataLoad(object):

    _all_data = None
    _extract_data_size = 0
    _class_num = 0

    def __init__(self, data_path, time_step, class_num):
        if not os.path.exists(data_path):
            print('%s is not found'%(data_path))
            raise FileExistsError
        self._time_step = time_step
        self._extract_data_size = self._time_step
        self._class_num = class_num
        self._data_file_list = [os.path.join(data_path, file) for file in os.listdir(data_path)]

        self._all_data = pd.DataFrame()
        for f in self._data_file_list:
            # 读取所有csv文件
            if 'csv' in f:
                data = pd.read_csv(f, index_col=False)
                self._all_data = self._all_data.append(data)

    def get_batch(self, batchsize, start_list=None):
        data_size = len(self._all_data.acc_x.values)

        if start_list is None:
            start_pos = [random.randint(1, data_size - self._extract_data_size) for _ in range(data_size)]
        else:
            if len(start_list) != batchsize:
                print('batchisze = ', batchsize)
                print('start_list length = ', len(start_list))
                raise KeyError('batchsize is no equal to start_list length!')
            start_pos = start_list

        train_x = []
        label_y = []
        for i in range(batchsize):

            train_x.append(self._all_data.iloc[start_pos[i]:start_pos[i]+self._extract_data_size, 0:3].values)
            label = [[0 for _ in range(self._class_num)] for _ in range(self._extract_data_size)]

            for s in range(self._extract_data_size):
                j = self._all_data.iloc[start_pos[i] + s:start_pos[i] + s + 1, 6].values[0]
                label[s][j] = 1
            label_y.append(label)

        return np.array(train_x), np.array(label_y)

    def get_test_data(self):
        """
        x shape = [datasize, 3]
        y shape = [datasize ,1]
        :return:
        """
        x = np.array(self._all_data.iloc[:, 0:3].values)
        y = np.array(self._all_data.iloc[:, 6].values)
        return x, y



if __name__ == '__main__':
    data = DataLoad('./dataset/train/', time_step=150, class_num=11)
    x, y = data.get_batch(50)
    print(x.shape)
    print(y.shape)
