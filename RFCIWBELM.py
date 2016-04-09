# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:30:42 2016

@author: pc
"""

from numpy import zeros, tanh, sqrt, random, power, sum, dot
from BELM import BELM
import time

class RFCIWBELM(BELM):
    
    def __init__(self, height, width, HidSize, OutSize, OPIUM_type, \
                 genWeights_type, data, label, q = 200, act_fun = tanh, \
                 H=1, stochastic=False, binaryTrain=False, binaryTest=True):
        
        print ">>> Initialing RF-CIW-BELM model <<<"
        smtic = time.time() 
        
        InSize = height*width
        
        if genWeights_type != 'rf-ciw':
             raise Exception("Wrong model called")
             
        BELM.__init__(self, InSize, HidSize, OutSize, OPIUM_type, \
                      genWeights_type, act_fun, H, \
                      stochastic, binaryTrain, binaryTest)
        
        data = self.__normalizeTest(data)
        
        N, _ = label.shape
        self.RandomWeight = zeros((self.HidSize, self.InSize))
        l = label.argmax(axis=1)
        start = 0
        end = 0
        for i in range(self.OutSize):
            start = end
            index = l == i
            Ni = sum(index)
            Mi = int(self.HidSize*float(Ni)/N)
            end += Mi
            if i == self.OutSize-1:
                Mi += self.HidSize-end
                end = self.HidSize
    
            Ri = random.randint(2, size = (Mi, Ni))
            Ri[Ri == 0] = -1
            Di = data[index]
            Wi = dot(Ri, Di)
            self.RandomWeight[start:end, :] = Wi
            
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
        print "RF-CIW-BELM Initialization complete, time cost: %3.2f seconds"%(smtoc-smtic) 

    def trainModel(self, train_data, train_label):
        train_data = self.__normalizeTest(train_data)
        super(RFCIWBELM, self).trainModel(train_data, train_label)
        
    def testModel(self, test_data, test_label):
        test_data = self.__normalizeTest(test_data)
        super(RFCIWBELM, self).testModel(test_data, test_label)
                
    def __normalizeTest(self, data):
        dataMean = data.mean(axis = 0)
        dataStd = data.std(axis = 0)
        m = dataMean > 0.001
        v = dataStd > 1.001
        if sum(m + v) != 0:
            print ">>>ATTENTION! Data unnormalized, Normalizing data..."
            dataStd[dataStd == 0] = 1
            data = (data - dataMean)/dataStd
            print ">>>Complete<<<"
        return data
        
    # excl is for data set trick, for mnist is 3
    def __genIndex(self, height, width, excl):
        uli = random.randint(0+excl, height-excl)
        ulj = random.randint(0+excl, width-excl)
        bri = random.randint(uli, height-excl)
        brj = random.randint(ulj, width-excl)
        return uli, ulj, bri, brj    