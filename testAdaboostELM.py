# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:43:27 2016

@author: zz
"""

from util import *
from AdaboostELM import AdaboostELM

train_data, train_label, test_data, test_label =  loadData(0.01)
feature_dim = train_data.shape[1]
label_dim = train_label.shape[1]

train_data = normalizeData(train_data)#/5000#int(feature_dim*1)
test_data = normalizeData(test_data)#/5000#int(feature_dim*1)
#train_label[train_label==1] = 250
#test_label[test_label==1] = 250
         
elm1Args = {}
elm1Args['height'] = 28
elm1Args['width'] = 28
#elm1Args['InSize'] = feature_dim
elm1Args['HidSize'] = int(feature_dim*10)
elm1Args['OutSize'] = label_dim
elm1Args['OPIUM_type'] = 'lite'
elm1Args['genWeights_type'] = 'rf-ciw'
#elm1Args['H'] = 0.25
#elm1Args['binaryTrain'] = False
#elm1Args['binaryTest'] = True
elm1Args['data'] = train_data
elm1Args['label'] = train_label

elm2Args = {}
elm2Args['height'] = 28
elm2Args['width'] = 28
#elm2Args['InSize'] = label_dim
elm2Args['HidSize'] = int(feature_dim*10)
elm2Args['OutSize'] = label_dim
elm2Args['OPIUM_type'] = 'lite'
elm2Args['genWeights_type'] = 'rf-c'
#elm2Args['H'] = 0.25
#elm2Args['binaryTrain'] = False
#elm2Args['binaryTest'] = True
elm2Args['data'] = train_data
#elm2Args['label'] = train_label

#elm3Args = {}
#elm3Args['height'] = 28
#elm3Args['width'] = 28
#elm3Args['InSize'] = feature_dim
#elm3Args['HidSize'] = int(feature_dim*10)
#elm3Args['OutSize'] = label_dim
#elm3Args['OPIUM_type'] = 'lite'
#elm3Args['genWeights_type'] = 'ciw'
#elm3Args['H'] = 0.25
#elm3Args['binaryTrain'] = False
#elm3Args['binaryTest'] = True
#elm3Args['data'] = train_data
#elm3Args['label'] = train_label

#elm4Args = {}
#elm4Args['height'] = 28
#elm4Args['width'] = 28
#elm4Args['InSize'] = feature_dim
#elm4Args['HidSize'] = int(feature_dim*10)
#elm4Args['OutSize'] = label_dim
#elm4Args['OPIUM_type'] = 'lite'
#elm4Args['genWeights_type'] = 'c'
#elm4Args['H'] = 0.25
#elm4Args['binaryTrain'] = False
#elm4Args['binaryTest'] = True
#elm4Args['data'] = train_data
#elm4Args['label'] = train_label

elm1 = {}
elm1['rf-ciw'] = elm1Args
elm2 = {}
elm2['rf-c'] = elm2Args
#elm3 = {}
#elm3['ciw'] = elm3Args
#elm4 = {}
#elm4['c'] = elm4Args

elmSet = [elm1, elm2, elm1, elm2] #, elm3, elm4]

adaboost_elm = AdaboostELM(elmSet)

adaboost_elm.trainModel(train_data, train_label)
#adaboost_elm.save(r"D:\workspace\Data\ELM\weights\adaboost")
adaboost_elm.testModel(test_data, test_label)
