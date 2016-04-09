# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 22:45:20 2016

@author: pc
"""

from numpy import zeros, tanh, dot, random, sqrt, power, sum, reshape, clip
from BELM import BELM
import pickle
import time

class CBELM(BELM):
    
    def __init__(self, InSize, HidSize, OutSize, OPIUM_type, genWeights_type, \
                 data, act_fun = tanh, H=1, stochastic=False,\
                 binaryTrain=False, binaryTest=True):
        
        print ">>> Initialing C-BELM model <<<"
        smtic = time.time() 
        
        if genWeights_type != 'c':
             raise Exception("Wrong model called")

        BELM.__init__(self, InSize, HidSize, OutSize, OPIUM_type, \
                      genWeights_type, act_fun, H, stochastic, \
                      binaryTrain, binaryTest)
        
        N = data.shape[0]
        
        self.RandomWeight = zeros((self.HidSize, self.InSize))
        self.b = zeros((self.HidSize, 1))
        for i in range(self.HidSize):
            d1 = data[random.randint(0, N), :]
            d2 = data[random.randint(0, N), :]  
            d = d1 + d2
            self.RandomWeight[i,:] = d/sqrt(sum(power(d, 2)))
            self.b[i] = random.uniform(0, 1)
        
        smtoc = time.time()
        print "C-BELM Initialization complete, time cost: %3.2f seconds"%(smtoc-smtic) 

    def train(self, train_item, train_label, alpha = 5): # training an ELM

        train_item = reshape(train_item, (self.InSize, 1))
        activation = dot(self.RandomWeight, train_item) + self.b # calculate activation
        activation = self.act_function(activation)

        if self.binaryTrain:
            # Binary Value Weights Training
            train_output_hat = dot(self.Wb, activation) # observed output
            e = reshape(train_label, (self.OutSize, 1)) - train_output_hat # deviation
            self.LinearWeight, self.Theta = self.train_fun(activation, e, self.LinearWeight, self.Theta, alpha)
            self.LinearWeight = clip(self.LinearWeight, -self.H, self.H)
            self.Wb = self.binarization(self.LinearWeight)
        else:
            # Real Value When Training
            train_output_hat = dot(self.LinearWeight, activation) # observed output
            e = reshape(train_label, (self.OutSize, 1)) - train_output_hat # deviation
            self.LinearWeight, self.Theta = self.train_fun(activation, e, self.LinearWeight, self.Theta, alpha)
            self.LinearWeight = clip(self.LinearWeight, -self.H, self.H)

    def recall(self, test_item): # testing examples with trained ELM
  
        test_item = reshape(test_item, (self.InSize,1))
        activation = self.act_function(dot(self.RandomWeight, test_item) + self.b) 

        if self.binaryTest:
            # Binary Weights When Testing
            self.Wb = self.binarization(self.LinearWeight)
            train_output_hat = dot(self.Wb, activation)                
        else:
            # Real Value Weights when Testing
            train_output_hat = dot(self.LinearWeight, activation)   

        return train_output_hat

    @property
    def getRandBias(self):
        return self.b
    
    def save(self, Filename):

        data_dict = { 'LinearWeight':self.LinearWeight,\
                      'RandomWeight':self.RandomWeight,\
                      'bias':self.b}
                    
        with open(Filename, 'wb') as f:
            pickle.dump(data_dict , f)
        
    def load(self, Filename):
        
        with open(Filename, 'rb') as f:
            data_dict = pickle.load(f)

        self.LinearWeight = data_dict['LinearWeight']
        self.RandomWeight = data_dict['RandomWeight']
        self.b = data_dict['bias']