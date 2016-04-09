# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:36:36 2016

@author: zz
"""

from RFCBELM import RFCBELM
from util import *

train_data, train_label, test_data, test_label =  loadData(0.1)
feature_dim = train_data.shape[1]
label_dim = train_label.shape[1]
 
train_data = normalizeData(train_data)#/5000#int(feature_dim*1)
test_data = normalizeData(test_data)#/5000#int(feature_dim*1)
train_label[train_label==1] = 250
test_label[test_label==1] = 250

rfcbelm = RFCBELM(28, 28, feature_dim*10, label_dim, 'lite', 'rf-c', train_data, \
                  H=0.25, binaryTrain=True, binaryTest=True)

rfcbelm.trainModel(train_data, train_label)
#rfcbelm.save(r"D:\workspace\Data\ELM\weights\rfcbelm")
rfcbelm.testModel(test_data, test_label)