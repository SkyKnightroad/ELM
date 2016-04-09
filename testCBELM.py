# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 23:00:17 2016

@author: pc
"""

from CBELM import CBELM
from util import *

train_data, train_label, test_data, test_label =  loadData(0.1)
feature_dim = train_data.shape[1]
label_dim = train_label.shape[1]
 
train_data = normalizeData(train_data)#/5000#int(feature_dim*1)
test_data = normalizeData(test_data)#/5000#int(feature_dim*1)
train_label[train_label==1] = 250
test_label[test_label==1] = 250

cbelm = CBELM(feature_dim, feature_dim*10, label_dim, 'lite', 'c', train_data, \
              H=0.25, binaryTrain=True, binaryTest=True)

cbelm.trainModel(train_data, train_label)
#cbelm.save(r"D:\workspace\Data\ELM\weights\cbelm")
cbelm.testModel(test_data, test_label)