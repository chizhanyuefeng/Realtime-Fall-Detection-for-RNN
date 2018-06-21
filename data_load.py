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

    def get_next_batch(self, batchsize):

        data_size = len(self._all_data.acc_x.values)

        train_x = []
        label_y = []
        for i in range(batchsize):
            start = random.randint(1, data_size-self._extract_data_size)
            train_x.append(self._all_data.iloc[start:start+self._extract_data_size, 0:3].values)
            label = [[0 for i in range(self._class_num)] for _ in range(self._extract_data_size)]

            for s in range(self._extract_data_size):
                j = self._all_data.iloc[start + s:start + s + 1, 6].values[0]
                label[s][j] = 1
            label_y.append(label)

        return np.array(train_x), np.array(label_y)


if __name__ == '__main__':
    data = DataLoad('./dataset/train/', time_step=10, class_num=2)
    # x,y = data.get_next_batch(1)
    # print(x)
    # print(y)
