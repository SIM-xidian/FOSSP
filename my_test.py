# import h5py
#
# # 打开.h5文件
# file = h5py.File('train.h5', 'r')
#
# # 读取数据
# keys = file.keys()
# print(keys)
# X = file['X'][:]
# Y = file['Y'][:]
# mode = file['mode'][:]
#
#
# # 关闭文件
# file.close()

import os

import natsort
import numpy as np
import pandas as pd

path = 'dataset/signal4classes'

X, Y = [], []
X_train, Y_train = [], []
X_test, Y_test = [], []

for c in sorted(os.listdir(path)):
    for filename in sorted(os.listdir(os.path.join(path, c))):
        file_path = os.path.join(os.path.join(path, c), filename)
        # 读取xlsx文件
        data = pd.read_excel(file_path)
        # 将数据转换为NumPy数组
        data_numpy = data.values.transpose(1, 0)
        X.append(data_numpy)
        Y.append(c)

X = np.array(X)
Y = np.array(Y)
print()
