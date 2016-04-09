# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:52:11 2016

@author: zz
"""

from util import *
from ELM import ELM

train_data, train_label, test_data, test_label =  loadData(0.1)
feature_dim = train_data.shape[1]
label_dim = train_label.shape[1]

elm = ELM(feature_dim, feature_dim*10, label_dim, 'lite', 'dec')

elm.trainModel(train_data, train_label)
#elm.save(r"D:\workspace\Data\ELM\weights\elm")
elm.testModel(test_data, test_label)