# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:49:18 2016

@author: zz
"""

from util import *
from BELM import BELM
    
train_data, train_label, test_data, test_label =  loadData(0.1)
feature_dim = train_data.shape[1]
label_dim = train_label.shape[1]

train_data = normalizeData(train_data)/5000#int(feature_dim*1)
test_data = normalizeData(test_data)/5000#int(feature_dim*1)
#train_label[train_label==1] = 250
#test_label[test_label==1] = 250

belm = BELM(feature_dim, feature_dim*10, label_dim, 'lite', 'dec', H=0.25, \
            binaryTrain=True, binaryTest=True)

belm.trainModel(train_data, train_label)
#belm.save(r"D:\workspace\Data\ELM\weights\belm")
belm.testModel(test_data, test_label)