import os
import random
import numpy as np
import pandas as pd

class LoadData(object):

    data_file_list = None
    current_file_index = 0
    extract_data_size = 0

    def __init__(self, data_path, time_step):
        if not os.path.exists(data_path):
            print('%s is not found'%(data_path))
            raise FileExistsError
        self.time_step = time_step
        self.extract_data_size = 50 * self.time_step
        self.data_file_list = [os.path.join(data_path, file) for file in os.listdir(data_path)]

    def get_next_batch(self, batchsize):
        data = pd.read_csv(self.data_file_list[self.current_file_index])
        self.current_file_index += 1
        data_size = len(data.acc_x.values)

        train_x = []
        for i in range(batchsize):
            start = random.randint(1, data_size-self.extract_data_size)
            train_x.append(data.iloc[start:start+self.extract_data_size, 0:6].values)
        print(np.array(train_x).shape)

if __name__ == '__main__':
    data = LoadData('./dataset/train/', time_step=3)
    data.get_next_batch(2)