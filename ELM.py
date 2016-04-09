# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:12:06 2016

@author: Hou Yimeng

modified by zz on Mar 15 2016
"""

""" building an ELM framework """

from AbstractNetwork import AbstractNetwork 
from numpy import zeros, tanh, dot, reshape, random, eye, sum
from OPIUM import *
import time
import pickle

class ELM(AbstractNetwork):
    
    def __init__(self, InSize, HidSize, OutSize, OPIUM_type, genWeights_type, act_fun = tanh):
        
        self.support_rand_type = {'bin', 'dec', 'ciw', 'c', 'rf', 'rf-ciw', 'rf-c'}
        
        print ">>> Initialing ELM model <<<"
        smtic = time.time() 
        
        AbstractNetwork.__init__(self, InSize, HidSize, OutSize)
        self.genWeights_type = genWeights_type # type of generating weights
        self.OPIUM_type = OPIUM_type # type of OPIUM algorithm
        self.RandomWeight = None
        
        if self.genWeights_type not in self.support_rand_type:
            raise Exception("Unsupported random weights distribution") 

        if self.genWeights_type == 'bin':
            self.RandomWeight = random.randint(2, size = (self.HidSize, self.InSize))
            self.RandomWeight[self.RandomWeight==0] = -1
        elif self.genWeights_type == 'dec':
            self.RandomWeight = 2*random.rand(self.HidSize, self.InSize) - 1                 
            
        self.LinearWeight = zeros((self.OutSize, self.HidSize))
        self.Theta = eye(self.HidSize)
        self.Alpha = 5
        
        if self.OPIUM_type == 'lite':
            self.Theta = 1
            self.train_fun = OPIUMl
        elif self.OPIUM_type == 'basic':
            self.train_fun = OPIUM
        elif self.OPIUM_type == 'dynamic':
            self.train_fun = OPIUMd
            self.Theta /= (self.Alpha**2)
        else:
            raise Exception("Unsupported OPIUM algorothm")
            
        self.act_function = act_fun
        
        smtoc = time.time()
        print "ELM Initialization complete, time cost: %3.2f seconds"%(smtoc-smtic) 
           
    def train(self, train_item, train_label, alpha = 5): # training an ELM

        train_item = reshape(train_item, (self.InSize, 1))
        activation = dot(self.RandomWeight, train_item) # calculate activation
        activation = self.act_function(activation)
        train_output_hat = dot(self.LinearWeight, activation) # observed output
        e = reshape(train_label, (self.OutSize, 1)) - train_output_hat # deviation
        
        self.LinearWeight, self.Theta = self.train_fun(activation, e, self.LinearWeight, self.Theta, alpha)

    def recall(self, test_item): # testing examples with trained ELM
  
        test_item = reshape(test_item, (self.InSize,1))
        activation = self.act_function(dot(self.RandomWeight, test_item))
        return dot(self.LinearWeight, activation)  
        
    def trainModel(self, train_data, train_label):
        print ">>>START TRAINING<<<"
        train_size = train_data.shape[0]
        show_time = int(train_size*0.1)
        start = time.time() 
        for i in range(train_size):
            self.train(train_data[i], train_label[i])
            if (i+1)  % show_time == 0:
                print 100.0*(i+1)/train_size, "% complete"
        end = time.time() 
        print "Training finished"
        print "Time cost %3.2f seconds"%(end - start)
        print "***************************************************************"
        
    def testModel(self, test_data, test_label):
        print ">>>START TESTING <<<"
        test_size, C = test_label.shape
        error_count = zeros(C)
        count = zeros(C)
        show_time = int(test_size*0.1)
        for i in range(test_size):
            predict = self.recall(test_data[i])
            label = test_label[i].argmax()
            count[label] += 1
            if predict.argmax() != label:
                error_count[label] += 1
            if (i+1)  % show_time == 0:
                print 100.0*(i+1)/test_size, "% complete"
        accuracy = 100.0*sum(error_count)/test_size
        print "Testing finished, error rate: ", accuracy, "%"
        print "Error Stastistics: "
        count[count==0] = 1
        accuracyE = 100.0*error_count/count
        index = [i for i, j in enumerate(accuracyE) if j>accuracy]
        print "Error rate for each class: "
        print accuracyE
        print "Number of tested data points for each class: "
        print count
        print "Class that error rate higher than overall error rate: "
        print index
        print "***************************************************************"

    @property
    def getLinearWeight(self):
        return self.LinearWeight

    @property
    def getRandWeight(self):
        return self.RandomWeight
        
    def save(self, Filename):

        data_dict = { 'LinearWeight':self.LinearWeight,\
                      'RandomWeight':self.RandomWeight}
                    
        with open(Filename, 'wb') as f:
            pickle.dump(data_dict , f)
        
    def load(self, Filename):
        
        with open(Filename, 'rb') as f:
            data_dict = pickle.load(f)

        self.LinearWeight = data_dict['LinearWeight']
        self.RandomWeight = data_dict['RandomWeight']