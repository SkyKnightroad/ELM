# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 17:23:36 2016

@author: zz
"""

from CELM import CELM
from util import *

train_data, train_label, test_data, test_label =  loadData(0.1)
feature_dim = train_data.shape[1]
label_dim = train_label.shape[1]
 
train_data = normalizeData(train_data)
test_data = normalizeData(test_data)

celm = CELM(feature_dim, int(feature_dim*10), label_dim, 'lite', 'c', train_data)

celm.trainModel(train_data, train_label)
#celm.save(r"D:\workspace\Data\ELM\weights\celm")
celm.testModel(test_data, test_label)