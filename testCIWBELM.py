# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 23:29:16 2016

@author: zz
"""

from CIWBELM import CIWBELM
from util import *

train_data, train_label, test_data, test_label =  loadData(0.1)
feature_dim = train_data.shape[1]
label_dim = train_label.shape[1]
 
train_data = normalizeData(train_data)#/5000#int(feature_dim*1)
test_data = normalizeData(test_data)#/5000#int(feature_dim*1)

train_label[train_label==1] = 250
test_label[test_label==1] = 250

ciwbelm = CIWBELM(feature_dim, int(feature_dim*10), label_dim, 'lite', 'ciw', \
                  train_data, train_label, H=0.25, \
                  binaryTrain=True, binaryTest=True)

ciwbelm.trainModel(train_data, train_label)
#ciwbelm.save(r"D:\workspace\Data\ELM\weights\ciwbelm")
ciwbelm.testModel(test_data, test_label)