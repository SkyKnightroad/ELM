# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:03:44 2016

@author: zz
"""

from numpy import zeros, tanh, dot, random, sqrt, power, sum
from ELM import ELM
import time

class CIWELM(ELM):
    
    def __init__(self, InSize, HidSize, OutSize, OPIUM_type, genWeights_type, data, label, act_fun = tanh):
        
        print ">>> Initialing CIW-ELM model <<<"
        smtic = time.time() 
        
        if genWeights_type != 'ciw':
             raise Exception("Wrong model called")
        
        ELM.__init__(self, InSize, HidSize, OutSize, OPIUM_type, genWeights_type, act_fun)
        
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
        self.RandomWeight /= sqrt(sum(power(self.RandomWeight, 2), axis=1)).reshape(self.HidSize, 1)
        
        smtoc = time.time()
        print "CIW-ELM Initialization complete, time cost: %3.2f seconds"%(smtoc-smtic) 
    
    def trainModel(self, train_data, train_label):
        train_data = self.__normalizeTest(train_data)
        super(CIWELM, self).trainModel(train_data, train_label)
        
    def testModel(self, test_data, test_label):
        test_data = self.__normalizeTest(test_data)
        super(CIWELM, self).testModel(test_data, test_label)
                
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