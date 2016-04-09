# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:39:59 2016

@author: zz
"""

from util import *
from MultiLayerELM import MultiLayerELM
#import numpy as np

train_data, train_label, test_data, test_label =  loadData(0.01)
feature_dim = train_data.shape[1]
label_dim = train_label.shape[1]

train_data = normalizeData(train_data)#/5000#int(feature_dim*1)
test_data = normalizeData(test_data)#/5000#int(feature_dim*1)
train_label[train_label==1] = 250
test_label[test_label==1] = 250
         
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
elm1Args['H'] = 0.25
elm1Args['randomPrec'] = 1
elm1Args['actPrec'] = 1
elm1Args['linearPrec'] = 1
elm1Args['callPrec'] = 1
elm1Args['fixedTrain'] = False
elm1Args['fixedTest'] = True

elm2Args = {}
elm2Args['height'] = 28
elm2Args['width'] = 28
#elm2Args['InSize'] = feature_dim
elm2Args['HidSize'] = int(feature_dim*10)
elm2Args['OutSize'] = label_dim
elm2Args['OPIUM_type'] = 'lite'
elm2Args['genWeights_type'] = 'rf-c'
#elm2Args['H'] = 0.25
#elm2Args['binaryTrain'] = False
#elm2Args['binaryTest'] = True
elm2Args['data'] = train_data
#elm2Args['label'] = train_label
elm2Args['H'] = 0.25
elm2Args['randomPrec'] = 1
elm2Args['actPrec'] = 1
elm2Args['linearPrec'] = 1
elm2Args['callPrec'] = 1
elm2Args['fixedTrain'] = False
elm2Args['fixedTest'] = True

elm1 = {}
elm1['fpga-rf-ciw-elm'] = elm1Args
elm2 = {}
elm2['fpga-rf-c-elm'] = elm2Args

l1 = [elm1, elm2]

elm3Args = {}
#elm3Args['height'] = 28
#elm3Args['width'] = 28
elm3Args['InSize'] = len(l1)*label_dim
elm3Args['HidSize'] = int(label_dim*50)
elm3Args['OutSize'] = label_dim
elm3Args['OPIUM_type'] = 'lite'
elm3Args['genWeights_type'] = 'dec'
#elm3Args['H'] = 0.25
#elm3Args['binaryTrain'] = False
#elm3Args['binaryTest'] = True
#elm3Args['data'] = train_data
#elm3Args['label'] = train_label

elm3 = {}
elm3['basic'] = elm3Args

l2 = [elm3]

ls = [l1, l2]

multi_layer_elm = MultiLayerELM(ls)

multi_layer_elm.trainModel(train_data, train_label)
#multi_layer_elm.save(r"D:\workspace\Data\ELM\weights\multi layer")
multi_layer_elm.testModel(test_data, test_label)

#multi_layer_elm.elmSet[0][0].testModel(test_data, test_label)
#multi_layer_elm.elmSet[0][1].testModel(test_data, test_label)

#predict1 = multi_layer_elm.modelRecall(multi_layer_elm.elmSet[0][0], test_data)
#predict = predict1.argmax(axis = 1)
#print "error rate: ", 100.0*sum(predict != test_label.argmax(axis=1))/test_label.shape[0]

#predict2 = multi_layer_elm.modelRecall(multi_layer_elm.elmSet[0][1], test_data)
#predict = predict2.argmax(axis = 1)
#print "error rate: ", 100.0*sum(predict != test_label.argmax(axis=1))/test_label.shape[0]

#comPredict = np.concatenate(np.array([predict1, predict2]) , axis=1)  
#predict3 = multi_layer_elm.modelRecall(multi_layer_elm.elmSet[1][0], comPredict)
#predict = predict3.argmax(axis = 1)
#print "error rate: ", 100.0*sum(predict != test_label.argmax(axis=1))/test_label.shape[0]


