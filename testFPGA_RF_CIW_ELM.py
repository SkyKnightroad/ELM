# -*- coding: utf-8 -*-
"""
Created on Tue Apr 05 23:48:50 2016

@author: zz
"""
#Input: XX.X
#Random Weights: X.X
#Activation: X.X
#Linear Weights: X.X (trained in real value)
# error rate:  7.9 %  [0.1 MNIST lite]

from util import loadData, normalizeData, around
from FPGA_RF_CIW_ELM import FPGA_RF_CIW_ELM
import numpy as np

prec = 3
    
train_data, train_label, test_data, test_label =  loadData(0.1)
feature_dim = train_data.shape[1]
label_dim = train_label.shape[1]

train_data = normalizeData(train_data)#/5000#int(feature_dim*1)
test_data = normalizeData(test_data)#/5000#int(feature_dim*1)
train_label[train_label==1] = 250
test_label[test_label==1] = 250
train_data = around(train_data, N_bits=prec)
test_data = around(test_data, N_bits=prec)

fpga_elm = FPGA_RF_CIW_ELM(28, 28, feature_dim*10, label_dim, 'lite', 'rf-ciw', train_data, train_label, \
                           H=0.25, randomPrec=3, actPrec=3, linearPrec=3, callPrec=3, fixedTrain = False, fixedTest=True)

print "Training data max dim:", np.max(train_data)
print "Training data min dim:", np.min(train_data)

print "Before Training  random max dim:", np.max(fpga_elm.RandomWeight)
print "Before Training  random min dim:", np.min(fpga_elm.RandomWeight)
print "Before Training  linear max dim:", np.max(fpga_elm.LinearWeight)
print "Before Training  linear min dim:", np.min(fpga_elm.LinearWeight)
print "Before Training  fixed point linear max dim:", np.max(fpga_elm.Wb)
print "Before Training  fixed point linear min dim:", np.min(fpga_elm.Wb)
print "_______________________________________________________________________"
fpga_elm.trainModel(train_data, train_label)
#fpga_elm.save(r"D:\workspace\Data\ELM\weights\fpga_rf_ciw_elm")
print "After Training  random max dim:", np.max(fpga_elm.RandomWeight)
print "After Training  random min dim:", np.min(fpga_elm.RandomWeight)
print "After Training  linear max dim:", np.max(fpga_elm.LinearWeight)
print "After Training  linear min dim:", np.min(fpga_elm.LinearWeight)
print "After Training  fixed point linear max dim:", np.max(fpga_elm.Wb)
print "After Training  fixed point linear min dim:", np.min(fpga_elm.Wb)
print "_______________________________________________________________________"
fpga_elm.testModel(test_data, test_label)