# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:06:21 2016

@author: zz
"""

from ELM import ELM
import numpy as np
import time

class BELM(ELM):
    
    def __init__(self, InSize, HidSize, OutSize, OPIUM_type, \
                 genWeights_type, act_fun = np.tanh, H=1, stochastic=False,\
                 binaryTrain=False, binaryTest=True):
        
        if not binaryTrain and not binaryTest:
            raise Exception("Wrong model called, please call ELM but NOT BELM")

        print ">>> Initialing B-ELM model <<<"
        smtic = time.time()
        
        ELM.__init__(self, InSize, HidSize, OutSize, OPIUM_type, genWeights_type, act_fun)
        self.H = H
        self.binaryTrain = binaryTrain
        self.binaryTest = binaryTest
        self.stochastic = stochastic
        self.LinearWeight = np.random.uniform(-self.H, self.H, size=(self.OutSize, self.HidSize))
        self.Wb = self.binarization(self.LinearWeight)
        
        smtoc = time.time()
        print "B-ELM Initialization complete, time cost: %3.2f seconds"%(smtoc-smtic) 
        
        if binaryTrain:
            print ">>>Training in binary fassion"
        else:
            print ">>>Training in real value fassion"
            
        if binaryTest:
            print ">>>Test in binary fassion"
        else:
            print ">>>Test in real value fassion"        

    def train(self, train_item, train_label, alpha = 5): # training an ELM
        
        train_item = np.reshape(train_item, (self.InSize, 1))
        activation = np.dot(self.RandomWeight, train_item) # calculate activation
        activation = self.act_function(activation)

        if self.binaryTrain:
            # Binary Value Weights Training
            train_output_hat = np.dot(self.Wb, activation) # observed output
            e = np.reshape(train_label, (self.OutSize, 1)) - train_output_hat # deviation
            self.LinearWeight, self.Theta = self.train_fun(activation, e, self.LinearWeight, self.Theta, alpha)
            self.LinearWeight = np.clip(self.LinearWeight, -self.H, self.H)
            self.Wb = self.binarization(self.LinearWeight)
        else:
            # Real Value When Training
            train_output_hat = np.dot(self.LinearWeight, activation) # observed output
            e = np.reshape(train_label, (self.OutSize, 1)) - train_output_hat # deviation
            self.LinearWeight, self.Theta = self.train_fun(activation, e, self.LinearWeight, self.Theta, alpha)
            self.LinearWeight = np.clip(self.LinearWeight, -self.H, self.H)   

    def recall(self, test_item): # testing examples with trained ELM
        
        test_item = np.reshape(test_item, (self.InSize,1))
        activation = self.act_function(np.dot(self.RandomWeight, test_item))  
        
        if self.binaryTest:
            # Binary Weights When Testing
            self.Wb = self.binarization(self.LinearWeight)
            train_output_hat = np.dot(self.Wb, activation)                
        else:
            # Real Value Weights when Testing
            train_output_hat = np.dot(self.LinearWeight, activation)   

        return train_output_hat

    def binarization(self, W):
        
        # [-1,1] -> [0,1]
        Wb = self.hard_sigmoid(W/self.H)
            
        # Stochastic BinaryConnect
        if self.stochastic:
            # print("stoch")
            Wb = np.random.binomial(n=1, p=Wb, size=Wb.shape)
        else:
            # Deterministic BinaryConnect (round to nearest)
            # print("det")
            Wb = np.round(Wb)
            
        # 0 or 1 -> -1 or 1
        Wb[Wb==1] = self.H
        Wb[Wb==0] = -self.H
    
        return Wb
    
    def hard_sigmoid(self, x):
        return np.clip((x+1.)/2.,0,1)
    