# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 12:17:34 2016

@author: zz
"""

from RFELM import RFELM
from util import *

train_data, train_label, test_data, test_label =  loadData(0.1)
feature_dim = train_data.shape[1]
label_dim = train_label.shape[1]
 
train_data = normalizeData(train_data)
test_data = normalizeData(test_data)

rfelm = RFELM(28, 28, feature_dim*10, label_dim, 'lite', 'rf', q = 200)

rfelm.trainModel(train_data, train_label)
#rfelm.save(r"D:\workspace\Data\ELM\weights\rfelm")
rfelm.testModel(test_data, test_label)