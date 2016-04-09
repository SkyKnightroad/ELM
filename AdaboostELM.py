# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:00:12 2016

@author: 
"""

import pickle
import numpy as np

class AdaboostELM(object):
    
    def __init__(self, argDictList):
        supportType = {'basic', 'ciw', 'c', 'rf', 'rf-ciw', 'rf-c', \
                       'binary', 'ciw-b','c-b', 'rf-b','rf-ciw-b', 'rf-c-b'}

        self.elmSet = []
        self.elmWeight = []
        self.elmNum = len(argDictList)
        self.argDictList = argDictList
        
        print ">>> Initializing cascade ELM models"
        
        for i in range(self.elmNum):
            print ">>Iitializing ", i+1, "th model"
            elmType = self.argDictList[i].keys()[0]                        
            if elmType not in supportType:
                raise Exception("Unsupport ELM type")
            kwarg = self.argDictList[i].values()[0]
        
            if elmType == 'basic':
                from ELM import ELM
                self.elmSet.append(ELM(**kwarg))
            elif elmType == 'ciw':
                from CIWELM import CIWELM
                self.elmSet.append(CIWELM(**kwarg))
            elif elmType == 'c':
                from CELM import CELM
                self.elmSet.append(CELM(**kwarg))
            elif elmType == 'rf':
                from RFELM import RFELM
                self.elmSet.append(RFELM(**kwarg))
            elif elmType == 'rf-ciw':
                from RFCIWELM import RFCIWELM
                self.elmSet.append(RFCIWELM(**kwarg))
            elif elmType == 'rf-c':
                from RFCELM import RFCELM
                self.elmSet.append(RFCELM(**kwarg))
            elif elmType == 'binary':
                from BELM import BELM
                self.elmSet.append(BELM(**kwarg))
            elif elmType == 'ciw-b':
                from CWIBELM import CIWBELM
                self.elmSet.append(CIWBELM(**kwarg))
            elif elmType == 'c-b':
                from CBELM import CBELM
                self.elmSet.append(CBELM(**kwarg))
            elif elmType == 'rf-b':
                from RFBELM import RFBELM
                self.elmSet.append(RFBELM(**kwarg))
            elif elmType == 'rf-ciw-b':
                from RFCIWBELM import RFCIWBELM
                self.elmSet.append(RFCIWBELM(**kwarg))
            elif elmType == 'rf-c-b':
                from RFCBELM import RFCBELM
                self.elmSet.append(RFCBELM(**kwarg))
                
        print ">>> Initialization complete"       
    
    def trainModel(self, train_data, train_label):
        print "Training start ..."
        M, N = train_label.shape
        A = np.zeros((self.elmNum, N))
        index = np.random.binomial(1, 0.2, M)
        valid_data = train_data[index==1, :]
        valid_label = train_label[index==1, :]
        train_data = train_data[index==0, :]
        train_label = train_label[index==0, :]
                
        for i in range(self.elmNum):
            print "Training ", i+1, "th model ..."
            self.elmSet[i].trainModel(train_data, train_label)
            
            print "Evaluating ", i+1, "th model performance ..."
            predict = self.modelRecall(self.elmSet[i], valid_data)
            correct = predict.argmax(axis = 1) == valid_label.argmax(axis = 1)
            A[i, :] = np.sum(valid_label[correct,:], axis = 0) / \
                            (1.0*np.sum(valid_label, axis = 0))
            
            print "Complete training", i+1, "th model"
        
        A = A / np.sum(A, axis = 0)
        self.elmWeight = [list(w) for w in A]    
            
    def testModel(self, test_data, test_label):
        
        print ">>>Test each component: <<<"
        for i in range(self.elmNum):
            print "____________Model ", i+1, "______________"
            print "Contribution of each class: "
            print self.elmWeight[i]
            self.elmSet[i].testModel(test_data, test_label)
            print ""

        print ">>>START TESTING<<<"
        test_size, C = test_label.shape
        error_count = np.zeros(C)
        count = np.zeros(C)
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
        
    def modelRecall(self, model, data):
        predict = []
        test_size = data.shape[0]
        for i in range(test_size):
            predict.append(model.recall(data[i]))
            
        predict = np.array(predict)
        M, N, _ = predict.shape
        predict = predict.reshape(M, N)
        
        return predict

    def recall(self, data_point):
 
        predictEst = 0        
        for i in range(self.elmNum):
            predict = self.elmSet[i].recall(data_point)
            predictEst += predict.T*self.elmWeight[i]
        
        return predictEst
        
    def save(self, path):
        
        for i in range(self.elmNum):
            elmType = self.argDictList[i].keys()[0]
            FileName = path + "\\" + elmType + "-" + str(i)
            
            data_dict = { 'LinearWeight':self.elmSet[i].LinearWeight,\
                          'RandomWeight':self.elmSet[i].RandomWeight, \
                          'RecallWeight': self.elmWeight[i]}
                              
            if elmType in {'c', 'rf-c', 'cb', 'rf-c-b'}:
                data_dict['bias'] = self.elmSet[i].b 
                
            with open(FileName, 'wb') as f:
                pickle.dump(data_dict , f)
        
    def load(self, path):

        for i in range(self.elmNum):
            elmType = self.argDictList[i].keys()[0]
            FileName = path + "\\" + elmType + "-" + str(i)
            
            with open(FileName, 'rb') as f:
                data_dict = pickle.load(f)            

            self.elmSet[i].LinearWeight = data_dict['LinearWeight']
            self.elmSet[i].RandomWeight = data_dict['RandomWeight'] 
            self.elmWeight[i] = data_dict['RecallWeight']
                
            if elmType in {'c', 'rf-c', 'cb', 'rf-c-b'}:
                self.elmSet[i].b = data_dict['bias']                 