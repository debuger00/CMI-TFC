# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:50:07 2021

@author: axmao2-c
"""

""" train and test dataset

author baiyu
"""

import random
import numpy as np
import torch

import platform
import pandas as pd
from sklearn.preprocessing import StandardScaler


#Data segmentation 数据分割
def Data_Segm(df_data, single=True, tri=False):
# 表示不会将样本组合成三个连续的片段，而是保持单一片段。如果 tri=True，则会将每三个连续片段组合成一个新的样本。
# 
  segments,counts = np.unique(df_data["segment"], return_counts = True)# 返回唯一的值，返回唯一的的时间
  samples = [] # 存储当前片段 s 中划分出来的所有样本。
  labels = [] #标签
  for s in segments: # 遍历所有的唯一的片段标识符
    data_segment = df_data[df_data['segment'] == s] #这一行代码从 df_data 中筛选出当前片段 s 的所有数据行，生成新的数据框 data_segment，只包含当前片段的所有数据
    sample_persegm = [] # 一个空列表，用于存储当前片段 s 中划分出来的所有样本。
    for j in range(0,len(data_segment),100): # 步长为100，overlap = 100 即50%
      # temp_sample = data_segment[['Ax','Ay','Az','Gx','Gy','Gz']].iloc[j:j+200,:].values
      temp_sample = data_segment[['Gx','Gy','Gz']].iloc[j:j+200,:].values #选取从第 j 行开始的 200 行数据（即一个长度为 200 的时间序列片段）。每个样本包含 200 个数据
      if len(temp_sample) == 200:
        sample_persegm.append(temp_sample)
    samples.append(sample_persegm)
    labels.append(list(set(data_segment['label']))[0])

  samples_all = []
  labels_all = []
  for i in range(len(labels)):
    if single:
      for s in samples[i]:
        samples_all.append([s])
        labels_all.append(labels[i])
    if tri:
      for j in range(len(samples[i])):
        if (j+2) < len(samples[i]):
          samples_all.append([samples[i][j], samples[i][j+1], samples[i][j+2]])
          labels_all.append(labels[i])
  
  return samples_all, labels_all

#Get training data, validation data, and test data
def get_data(train_subjects = [2,3,7,8], valid_subject = 11, test_subject = 14):
    
    path = 'C:\\Workplace\\Data\\Acc_Gyr_Data.csv'
    if platform.system()=='Linux': 
        path = '/home/axmao2/data/equine/Acc_Gyr_Data.csv'
    df_train_raw = pd.read_csv(path)
    df_train_raw = df_train_raw.drop(['sample_index'], axis=1) # 删除 sample_index 的列，axis=1表示删除的是列不是行

    #数值对应6中行为['eating', 'galloping', 'standing', 'trotting', 'walking-natural', 'walking-rider'] 数字代替字符串
    df_train_raw.replace({'grazing':0,'eating':0,'galloping-natural':1,'galloping-rider':1,'standing':2,'trotting-rider':3,'trotting-natural':3,'walking-natural':4,'walking-rider':5},inplace = True)  
    #class_labels = ['eating', 'galloping', 'standing', 'trotting', 'walking-natural', 'walking-rider']
    feature_columns = df_train_raw.columns[0:6]  # 提取 df_train_raw 数据框的前六列，并将这些列的列名赋值给 feature_columns 变量
    
    #data standardization2
    for i in feature_columns: # 对每一列进行遍历
        s_raw = StandardScaler().fit_transform(df_train_raw[i].values.reshape(-1,1))
        df_train_raw[i]  = s_raw.reshape(-1) # 把每一行数据都进行归一化
        
    #get the training data, validation data, and test data
    df_train = df_train_raw.loc[df_train_raw['subject'].isin(train_subjects)] # 看定义 train_subjects = [2,3,7,8] 属于这个的都属于训练集
    df_valid = df_train_raw[df_train_raw['subject']==valid_subject] # 同上
    df_test = df_train_raw[df_train_raw['subject']==test_subject]
    
    return df_train, df_valid, df_test


df_train, df_valid, df_test = get_data(train_subjects = [14,2,3,7], valid_subject = 8, test_subject = 11)
samples_train, labels_train = Data_Segm(df_train, single=True, tri=False)
samples_valid, labels_valid = Data_Segm(df_valid, single=True, tri=False)
samples_test, labels_test = Data_Segm(df_test, single=True, tri=False)

tensor_samples_train = torch.from_numpy(np.array(samples_train)).float()
tensor_label_train = torch.from_numpy(np.array(labels_train)).type(torch.LongTensor)

tensor_samples_valid = torch.from_numpy(np.array(samples_valid)).float()
tensor_label_valid = torch.from_numpy(np.array(labels_valid)).type(torch.LongTensor)

tensor_samples_test = torch.from_numpy(np.array(samples_test)).float()
tensor_label_test = torch.from_numpy(np.array(labels_test)).type(torch.LongTensor)

torch.save([tensor_samples_train, tensor_samples_valid, tensor_samples_test, tensor_label_train, tensor_label_valid, tensor_label_test], "./myTensor_Gyr_6.pt")





