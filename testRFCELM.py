# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 00:21:19 2016

@author: zz
"""

from RFCELM import RFCELM
from util import *

train_data, train_label, test_data, test_label =  loadData(0.1)
feature_dim = train_data.shape[1]
label_dim = train_label.shape[1]
 
train_data = normalizeData(train_data)
test_data = normalizeData(test_data)

rfcelm = RFCELM(28, 28, feature_dim*10, label_dim, 'lite', 'rf-c', train_data)

rfcelm.trainModel(train_data, train_label)
#rfcelm.save(r"D:\workspace\Data\ELM\weights\rfcelm")
rfcelm.testModel(test_data, test_label)