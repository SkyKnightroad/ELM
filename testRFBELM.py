# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 00:01:08 2016

@author: pc
"""

from RFBELM import RFBELM
from util import *

train_data, train_label, test_data, test_label =  loadData(0.1)
feature_dim = train_data.shape[1]
label_dim = train_label.shape[1]
 
train_data = normalizeData(train_data)#/5000#int(feature_dim*1)
test_data = normalizeData(test_data)#/5000#int(feature_dim*1)
train_label[train_label==1] = 250
test_label[test_label==1] = 250

rfbelm = RFBELM(28, 28, feature_dim*10, label_dim, 'lite', 'rf', q=200, \
                H=0.25, binaryTrain=True, binaryTest=True)

rfbelm.trainModel(train_data, train_label)
#rfbelm.save(r"D:\workspace\Data\ELM\weights\rfbelm")
rfbelm.testModel(test_data, test_label)