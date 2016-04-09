# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:48:13 2016

@author: pc
"""

from RFCIWBELM import RFCIWBELM
from util import *

train_data, train_label, test_data, test_label =  loadData(0.1)
feature_dim = train_data.shape[1]
label_dim = train_label.shape[1]
 
train_data = normalizeData(train_data)#/5000#int(feature_dim*1)
test_data = normalizeData(test_data)#/5000#int(feature_dim*1)
train_label[train_label==1] = 250
test_label[test_label==1] = 250

rfciwbelm = RFCIWBELM(28, 28, int(feature_dim*10), label_dim, 'lite', 'rf-ciw',\
                      train_data, train_label, H=0.25, binaryTrain=True, binaryTest=True)

rfciwbelm.trainModel(train_data, train_label)
#rfciwbelm.save(r"D:\workspace\Data\ELM\weights\rfciwbelm")
rfciwbelm.testModel(test_data, test_label)