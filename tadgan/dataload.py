import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

import warnings
warnings.filterwarnings("ignore")

class SWATLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = MinMaxScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data.replace('', np.nan, inplace=True)
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.values[:, 1:]
        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')
        test_data.replace('', np.nan, inplace=True)
        test_data = test_data.apply(pd.to_numeric, errors='coerce')
        test_data = test_data.values[:, 1:]

        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.7):]
        self.train = self.train[:(int)(data_len * 0.7)]
        self.test_labels = pd.read_csv(data_path + '/test_labels.csv').values

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print("test_labels:", self.test_labels.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



class TONLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = MinMaxScaler()
        data = pd.read_csv(data_path + '/train_data.csv')
        data = data.iloc[1:].replace("",0).fillna(0)
        data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
        data = data.values[:, 1:]
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_data = pd.read_csv(data_path + '/test_data.csv')
        test_data = test_data.iloc[1:].replace("",0).fillna(0)
        test_data = test_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        test_data = test_data.values[:, 1:]
        self.test = self.scaler.transform(test_data)

        # 训练集分割
        self.train = data
        data_len = len(self.train)
        split = int(0.3*data_len)
        indices = list(range(data_len))
        #np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        self.val = self.train[val_indices]
        self.train = self.train[train_indices]
        self.test_labels = pd.read_csv(data_path + '/test_labels.csv')

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print("test_labels:", self.test_labels.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = MinMaxScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.train = self.train[:(int)(data_len * 0.8)]
        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = MinMaxScaler()
        
        # Load training data from .npy file
        data = np.load(os.path.join(data_path, "T-9_train.npy"))
        data = data.reshape((-1, 55))  # Ensure the shape is correct
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        # Load test data from .npy file
        test_data = np.load(os.path.join(data_path, "T-9_test.npy"))
        test_data = test_data.reshape((-1, 55))  # Ensure the shape is correct
        self.test = self.scaler.transform(test_data)
        
        test_len = len(self.test)
        #self.test = self.test[:int(0.5 * test_len)]
        self.train = data
        data_len = len(self.train)
        split = int(0.3 * data_len)
        indices = list(range(data_len))
        # np.random.shuffle(indices)  # Uncomment if shuffling is needed
        train_indices, val_indices = indices[split:], indices[:split]
        self.val = self.train[val_indices]
        self.train = self.train[train_indices]
        
        # Load test labels from .npy file
        self.test_labels = np.load(os.path.join(data_path, "T-9_labels.npy"))
        #self.test_labels = self.test_labels[:int(0.5 * test_len)]

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = MinMaxScaler()
        
        # Load training data from .npy file
        data = np.load(os.path.join(data_path, "A-6_train.npy"))
        data = data.reshape((-1, 25))  # Ensure the shape is correct
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        # Load test data from .npy file
        test_data = np.load(os.path.join(data_path, "A-6_test.npy"))
        test_data = test_data.reshape((-1, 25))  # Ensure the shape is correct
        self.test = self.scaler.transform(test_data)
        
        test_len = len(self.test)
        #self.test = self.test[:int(0.5 * test_len)]
        self.train = data
        data_len = len(self.train)
        split = int(0.3 * data_len)
        indices = list(range(data_len))
        # np.random.shuffle(indices)  # Uncomment if shuffling is needed
        train_indices, val_indices = indices[split:], indices[:split]
        self.val = self.train[val_indices]
        self.train = self.train[train_indices]
        
        # Load test labels from .npy file
        self.test_labels = np.load(os.path.join(data_path, "A-6_labels.npy"))
        #self.test_labels = self.test_labels[:int(0.5 * test_len)]

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class NSL_KDDLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = MinMaxScaler()
        data = pd.read_csv(data_path + '/KDDtrain+.txt',header=None)
        data.columns = range(data.shape[1])
        data = data.drop(columns=[41])
        data = data.drop(columns=[42])
        data[1] = le.fit_transform(data[1])
        data[2] = le.fit_transform(data[2])
        data[3] = le.fit_transform(data[3])
        data = data.values[:, :]
        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/KDDtest+.txt',header=None)
        test_data.columns = range(test_data.shape[1])
        test_label = test_data[41]
        test_data = test_data.drop(columns=[41])
        test_data = test_data.drop(columns=[42])
        test_data[1] = le.fit_transform(test_data[1])
        test_data[2] = le.fit_transform(test_data[2])
        test_data[3] = le.fit_transform(test_data[3])
        test_data = test_data.values[:, :]
        test_data = np.nan_to_num(test_data)
        test_label = (test_label != 'normal')


        self.test = self.scaler.transform(test_data)

        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.train = self.train[:(int)(data_len * 0.8)]

        self.test_labels = test_label.values

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = MinMaxScaler()
        
        # 加载训练数据
        f = open(os.path.join(data_path, "SMD_train.pkl"), "rb")
        data = pickle.load(f).reshape((-1, 38))[0:None,:]
        f.close()
        if np.any(np.isnan(data)):
            data = np.nan_to_num(data)
        
        # 数据标准化
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        # 加载测试数据
        f = open(os.path.join(data_path, "SMD_test.pkl"), "rb")
        test_data = pickle.load(f).reshape((-1, 38))[0:None,:]
        f.close()
        if np.any(np.isnan(test_data)):
            test_data = np.nan_to_num(test_data) 
        self.test = self.scaler.transform(test_data)
        
        # 训练集分割
        self.train = data
        data_len = len(self.train)
        split = int(0.3*data_len)
        indices = list(range(data_len))
        #np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        self.val = self.train[val_indices]
        self.train = self.train[train_indices]
        
        # 加载测试标签
        f = open(os.path.join(data_path, "SMD_test_label.pkl"), "rb")
        self.test_labels = pickle.load(f).reshape((-1))[0:None]
        f.close()
        
        # 创建训练和验证集的标签（正常样本标签为0）
        self.train_labels = np.zeros(len(self.train))
        self.val_labels = np.zeros(len(self.val))

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.train_labels[index:index + self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.val_labels[index:index + self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



def get_loader_segment(data_path, batch_size, win_size=20, step=10, mode='train', dataset='KDD'):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'KDD'):
        dataset = NSL_KDDLoader(data_path, win_size, step, mode)
    elif (dataset == 'TON'):
        dataset = TONLoader(data_path, win_size, step, mode)
    elif(dataset == 'SWAT'):
        dataset = SWATLoader(data_path, win_size, 10, mode)

    shuffle = False
    if mode == 'train':
        shuffle = False

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
