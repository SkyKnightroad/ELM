# -*- coding: utf-8 -*-
"""
Created on Tue Apr 05 23:41:38 2016

@author: zz
"""

#from bitstring import Bits
from util import around
from ELM import ELM
import numpy as np
import time

class FPGA_RF_CIW_ELM(ELM):
    
    def __init__(self, height, width, HidSize, OutSize, OPIUM_type, \
                 genWeights_type, data, label, q = 200, act_fun = np.tanh, H=1,\
                 randomPrec=3, actPrec=3, linearPrec=3, callPrec=3, fixedTrain = True, fixedTest=True):

        if genWeights_type != 'rf-ciw':
             raise Exception("Wrong model called")

        print ">>> Initialing FPGA-RF-CIW-ELM model <<<"
        smtic = time.time()
        
        InSize = height*width 
        
        ELM.__init__(self, InSize, HidSize, OutSize, OPIUM_type, genWeights_type, act_fun)
        self.H = H
        self.randomPrec = randomPrec
        self.actPrec = actPrec
        self.linearPrec = linearPrec
        self.callPrec = callPrec
        self.fixedTrain = fixedTrain
        self.fixedTest = fixedTest
        #self.LinearWeight = np.random.uniform(-self.H, self.H, size=(self.OutSize, self.HidSize))
        
        #data = self.__normalizeTest(data)
        
        N = label.shape[0]
        self.RandomWeight = np.zeros((self.HidSize, self.InSize))
        l = label.argmax(axis=1)
        start = 0
        end = 0
        for i in range(self.OutSize):
            start = end
            index = l == i
            Ni = np.sum(index)
            Mi = int(self.HidSize*float(Ni)/N)
            end += Mi
            if i == self.OutSize-1:
                Mi += self.HidSize-end
                end = self.HidSize
    
            Ri = np.random.randint(2, size = (Mi, Ni))
            Ri[Ri == 0] = -1
            Di = data[index]
            Wi = np.dot(Ri, Di)
            self.RandomWeight[start:end, :] = Wi
            
        F = np.zeros((self.HidSize, self.InSize))
        
        for i in range(self.HidSize):
            uli, ulj, bri, brj = self.__genIndex(height, width, 3)
            size = (bri + 1 - uli)*(brj + 1 - ulj)
            while size  < q:
                uli, ulj, bri, brj = self.__genIndex(height, width, 3)
                size = (bri + 1 - uli)*(brj + 1 - ulj)
            Fi = np.zeros((height, width))
            Fi[uli:bri+1, ulj:brj+1] = 1
            F[i, :] = Fi.flatten()
        
        self.RandomWeight *= F     
        self.RandomWeight /= np.sqrt(np.sum(np.power(self.RandomWeight, 2), \
                                     axis=1)).reshape(self.HidSize, 1)    
        
        if self.randomPrec == 0:
            print ">>>Binary random weights"
            self.RandomWeight = self.binarization(self.RandomWeight)
        else:
            print ">>>Fixed point precise random weights: ", self.randomPrec, "bits representation"
            self.RandomWeight = around(self.RandomWeight, N_bits=self.randomPrec)
        
        if self.linearPrec == 0:
            print ">>>Binary linear weights"
            self.Wb = self.binarization(self.LinearWeight)
        else:
            print ">>>Fixed point precise linear weights: ", self.linearPrec, "bits representation"
            self.Wb = around(self.LinearWeight, N_bits=self.linearPrec)
            
        if self.fixedTrain:
            print ">>>Fixed point precise linear weight will be used in training\
                   phase: ", self.linearPrec, "bits representation"      
        else:
            print ">>>Real value linear weights will be used in training phase"
                   
        if self.fixedTest:
            print ">>>Linear weights will be fixed point precise weights \
                    in testing phase: ", self.linearPrec, "bits representation" 
        else:
            print ">>>Linear weights will be real value weights in testing phase"      
        
        smtoc = time.time()
        print "FPGA-RF-CIW-ELM Initialization complete, time cost: %3.2f seconds"%(smtoc-smtic) 
    
    def train(self, train_item, train_label, alpha = 5): # training an ELM
        
        train_item = np.reshape(train_item, (self.InSize, 1))
        activation = np.dot(self.RandomWeight, train_item) # calculate activation
        activation = around(activation, N_bits=self.actPrec)
        activation = self.act_function(activation)
        activation = around(activation, N_bits=self.actPrec)
        #print "Training Activation max dim:", np.max(activation)
        #print "Training Activation min dim:", np.min(activation)
        if self.fixedTrain:
            # Fixed point precise Weights Training
            train_output_hat = np.dot(self.Wb, activation) # observed output
        else:
            # Real Value When Training
            train_output_hat = np.dot(self.LinearWeight, activation) # observed output
        train_output_hat = around(train_output_hat, N_bits=self.callPrec)
        #print "Training call max dim:", np.max(train_output_hat)
        #print "Training call min dim:", np.min(train_output_hat)
        
        e = np.reshape(train_label, (self.OutSize, 1)) - train_output_hat # deviation
        self.LinearWeight, self.Theta = self.train_fun(activation, e, self.LinearWeight, self.Theta, alpha)
        if self.linearPrec == 0:
            self.LinearWeight = np.clip(self.LinearWeight, -self.H, self.H)
            self.Wb = self.binarization(self.LinearWeight)
        else:
            self.Wb = around(self.LinearWeight, N_bits=self.linearPrec)
            
    def recall(self, test_item): # testing examples with trained ELM
        
        test_item = np.reshape(test_item, (self.InSize,1))
        activation = np.dot(self.RandomWeight, test_item)
        activation = around(activation, N_bits=self.actPrec)
        activation = self.act_function(activation) 
        activation = around(activation, N_bits=self.actPrec)
        #print "Testing Activation max dim:", np.max(activation)
        #print "Testing Activation min dim:", np.min(activation)        
        if self.fixedTest:
            # Fixed point precise Weights When Testing
            test_output_hat = np.dot(self.Wb, activation)                
        else:
            # Real Value Weights when Testing
            test_output_hat = np.dot(self.LinearWeight, activation)  
        test_output_hat = around(test_output_hat, N_bits=self.callPrec)
        #print "Testing call max dim:", np.max(test_output_hat)
        #print "Testing call min dim:", np.min(test_output_hat)
        return test_output_hat
        
    def binarization(self, W):
        
        Wb = self.hard_sigmoid(W/self.H)
        
        Wb = np.round(Wb)
            
        Wb[Wb==1] = self.H
        Wb[Wb==0] = -self.H
    
        return Wb
    
    def hard_sigmoid(self, x):
        return np.clip((x+1.)/2.,0,1)
        
    # excl is for data set trick, for mnist is 3
    def __genIndex(self, height, width, excl):
        uli = np.random.randint(0+excl, height-excl)
        ulj = np.random.randint(0+excl, width-excl)
        bri = np.random.randint(uli, height-excl)
        brj = np.random.randint(ulj, width-excl)
        return uli, ulj, bri, brj 
