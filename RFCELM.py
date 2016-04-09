# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 00:09:57 2016

@author: zz
"""

from numpy import zeros, tanh, random, sqrt, power, sum, reshape, dot
from ELM import ELM
import pickle
import time

class RFCELM(ELM):
    
    def __init__(self, height, width, HidSize, OutSize, OPIUM_type, \
                 genWeights_type, data, q = 200, act_fun = tanh):
        
        print ">>> Initialing RF-C-ELM model <<<"
        smtic = time.time() 

        InSize = height*width  
        
        if genWeights_type != 'rf-c':
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
        
        F = zeros((self.HidSize, self.InSize))
        
        for i in range(self.HidSize):
            uli, ulj, bri, brj = self.__genIndex(height, width, 3)
            size = (bri + 1 - uli)*(brj + 1 - ulj)
            while size  < q:
                uli, ulj, bri, brj = self.__genIndex(height, width, 3)
                size = (bri + 1 - uli)*(brj + 1 - ulj)
            Fi = zeros((height, width))
            Fi[uli:bri+1, ulj:brj+1] = 1
            F[i, :] = Fi.flatten()
        
        self.RandomWeight *= F     
        self.RandomWeight /= sqrt(sum(power(self.RandomWeight, 2), axis=1)).reshape(self.HidSize, 1)             
        
        smtoc = time.time()
        print "RF-C-ELM Initialization complete, time cost: %3.2f seconds"%(smtoc-smtic) 

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

    # excl is for data set trick, for mnist is 3
    def __genIndex(self, height, width, excl):
        uli = random.randint(0+excl, height-excl)
        ulj = random.randint(0+excl, width-excl)
        bri = random.randint(uli, height-excl)
        brj = random.randint(ulj, width-excl)
        return uli, ulj, bri, brj