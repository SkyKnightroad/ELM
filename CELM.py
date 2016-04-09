# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:45:00 2016

@author: zz
"""

from numpy import zeros, tanh, dot, random, sqrt, power, sum, reshape
from ELM import ELM
import pickle
import time

class CELM(ELM):
    
    def __init__(self, InSize, HidSize, OutSize, OPIUM_type, genWeights_type, data, act_fun = tanh):
        
        print ">>> Initialing C-ELM model <<<"
        smtic = time.time() 
        
        if genWeights_type != 'c':
             raise Exception("Wrong model called")

        ELM.__init__(self, InSize, HidSize, OutSize, OPIUM_type, genWeights_type, act_fun)
        
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
        print "C-ELM Initialization complete, time cost: %3.2f seconds"%(smtoc-smtic) 

    def train(self, train_item, train_label, alpha = 5): # training an ELM

        train_item = reshape(train_item, (self.InSize, 1))
        activation = dot(self.RandomWeight, train_item) + self.b # calculate activation
        activation = self.act_function(activation)
        train_output_hat = dot(self.LinearWeight, activation) # observed output
        e = reshape(train_label, (self.OutSize, 1)) - train_output_hat # deviation
        
        self.LinearWeight, self.Theta = self.train_fun(activation, e, self.LinearWeight, self.Theta, alpha)

    def recall(self, test_item): # testing examples with trained ELM
  
        test_item = reshape(test_item, (self.InSize,1))
        activation = self.act_function(dot(self.RandomWeight, test_item) + self.b) 
        return dot(self.LinearWeight, activation)  

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