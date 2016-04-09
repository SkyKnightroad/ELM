# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 00:02:08 2016

@author: zz
"""

from RFCIWELM import RFCIWELM
from util import *

train_data, train_label, test_data, test_label =  loadData(0.1)
feature_dim = train_data.shape[1]
label_dim = train_label.shape[1]
 
train_data = normalizeData(train_data)
test_data = normalizeData(test_data)

rfciwelm = RFCIWELM(28, 28, feature_dim*10, label_dim, 'lite', 'rf-ciw', train_data, train_label)

rfciwelm.trainModel(train_data, train_label)
#rfciwelm.save(r"D:\workspace\Data\ELM\weights\rfciwelm")
rfciwelm.testModel(test_data, test_label)