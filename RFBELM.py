# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 23:54:49 2016

@author: zz
"""


from numpy import zeros, tanh, random, sqrt, power, sum
from BELM import BELM
import time

class RFBELM(BELM):
    
    def __init__(self, height, width, HidSize, OutSize, OPIUM_type, \
                 genWeights_type, q = 200, act_fun = tanh, \
                 H=1, stochastic=False,\
                 binaryTrain=False, binaryTest=True):
        
        print ">>> Initialing RF-BELM model <<<"
        smtic = time.time() 

        InSize = height*width  
        
        if genWeights_type != 'rf':
             raise Exception("Wrong model called")
        
        BELM.__init__(self, InSize, HidSize, OutSize, OPIUM_type, genWeights_type, \
                      act_fun, H, stochastic, binaryTrain, binaryTest)
        
        self.RandomWeight = 2*random.rand(self.HidSize, self.InSize) - 1  
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
        print "RF-BELM Initialization complete, time cost: %3.2f seconds"%(smtoc-smtic) 
     
    # excl is for data set trick, for mnist is 3
    def __genIndex(self, height, width, excl):
        uli = random.randint(0+excl, height-excl)
        ulj = random.randint(0+excl, width-excl)
        bri = random.randint(uli, height-excl)
        brj = random.randint(ulj, width-excl)
        return uli, ulj, bri, brj