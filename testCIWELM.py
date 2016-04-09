# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 23:31:01 2016

@author: zz
"""

from CIWELM import CIWELM
from util import *

train_data, train_label, test_data, test_label =  loadData(0.01)
feature_dim = train_data.shape[1]
label_dim = train_label.shape[1]
 
train_data = normalizeData(train_data)
test_data = normalizeData(test_data)

ciwelm = CIWELM(feature_dim, feature_dim*10, label_dim, 'lite', 'ciw', train_data, train_label)

ciwelm.trainModel(train_data, train_label)
#ciwelm.save(r"D:\workspace\Data\ELM\weights\ciwelm")
ciwelm.testModel(test_data, test_label)