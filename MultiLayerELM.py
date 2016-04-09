# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:33:50 2016

@author: zz
"""

import numpy as np
import pickle

class MultiLayerELM(object):
    
    def __init__(self, argDictList):
        supportType = {'basic', 'ciw', 'c', 'rf', 'rf-ciw', 'rf-c', \
                       'binary', 'ciw-b','c-b', 'rf-b','rf-ciw-b', 'rf-c-b',\
                       'fpga-rf-ciw-elm', 'fpga-rf-c-elm'}
        
        self.argDictList = argDictList
        self.elmSet = []
        self.layerNum = len(argDictList)
        self.elmNums = []
        for i in range(self.layerNum):
            self.elmNums.append(len(self.argDictList[i]))
        
        print ">>> Initializing cascade ELM models"
        
        for i in range(self.layerNum):
            elmLayer = []
            for j in range(self.elmNums[i]):
                print ">>Iitializing ", i+1, "layer", j+1, "th model"
                elmType = self.argDictList[i][j].keys()[0]                        
                if elmType not in supportType:
                    raise Exception("Unsupport ELM type")
                kwarg = self.argDictList[i][j].values()[0]
        
                if elmType == 'basic':
                    from ELM import ELM
                    elmLayer.append(ELM(**kwarg))
                elif elmType == 'ciw':
                    from CIWELM import CIWELM
                    elmLayer.append(CIWELM(**kwarg))
                elif elmType == 'c':
                    from CELM import CELM
                    elmLayer.append(CELM(**kwarg))
                elif elmType == 'rf':
                    from RFELM import RFELM
                    elmLayer.append(RFELM(**kwarg))
                elif elmType == 'rf-ciw':
                    from RFCIWELM import RFCIWELM
                    elmLayer.append(RFCIWELM(**kwarg))
                elif elmType == 'rf-c':
                    from RFCELM import RFCELM
                    elmLayer.append(RFCELM(**kwarg))
                elif elmType == 'binary':
                    from BELM import BELM
                    elmLayer.append(BELM(**kwarg))
                elif elmType == 'ciw-b':
                    from CWIBELM import CIWBELM
                    elmLayer.append(CIWBELM(**kwarg))
                elif elmType == 'c-b':
                    from CBELM import CBELM
                    elmLayer.append(CBELM(**kwarg))
                elif elmType == 'rf-b':
                    from RFBELM import RFBELM
                    elmLayer.append(RFBELM(**kwarg))
                elif elmType == 'rf-ciw-b':
                    from RFCIWBELM import RFCIWBELM
                    elmLayer.append(RFCIWBELM(**kwarg))
                elif elmType == 'rf-c-b':
                    from RFCBELM import RFCBELM
                    elmLayer.append(RFCBELM(**kwarg))
                elif elmType == 'fpga-rf-ciw-elm':
                    from FPGA_RF_CIW_ELM import FPGA_RF_CIW_ELM
                    elmLayer.append(FPGA_RF_CIW_ELM(**kwarg))
                elif elmType == 'fpga-rf-c-elm':
                    from FPGA_RF_C_ELM import FPGA_RF_C_ELM
                    elmLayer.append(FPGA_RF_C_ELM(**kwarg))                      
                    
            self.elmSet.append(elmLayer)
                
        print ">>> Initialization complete"       
    
    def trainModel(self, train_data, train_label):
        print ">>> Training strat, please be patient"
        elmInput = train_data
        for i in range(self.layerNum):
            print "Strat training ", i+1, "th layer"
            predict = []
            for j in range(self.elmNums[i]):
                print "Training layer", i+1, j+1, "th model"
                self.elmSet[i][j].trainModel(elmInput, train_label)
                print "Complete training layer", i+1, j+1, "th model"
                if i < self.layerNum - 1:
                    print "Generate input for next layer"
                    predict.append(self.modelRecall(self.elmSet[i][j], elmInput))
            print "Complete training ", i+1, "th layer" 
            if i < self.layerNum - 1:
                elmInput = np.concatenate(np.array(predict) , axis=1)
        print ">>>Training complete, thank you for your patient"

    def testModel(self, test_data, test_label):
        print ">>>START TESTING <<<"
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
        modelInput = data_point
        for i in range(self.layerNum):
            predict = []
            for j in range(self.elmNums[i]):
                predict.append(self.elmSet[i][j].recall(modelInput))
            modelInput = np.concatenate(np.array(predict) , axis=0)  
        
        return predict[0]     
    
    def save(self, path):
        
        for i in range(self.layerNum):
            for j in range(self.elmNums[i]):
                elmType = self.argDictList[i][j].keys()[0]
                FileName = path + "\\" + elmType + "-" + str(i) + "-" + str(j)
            
                data_dict = { 'LinearWeight':self.elmSet[i][j].LinearWeight,\
                              'RandomWeight':self.elmSet[i][j].RandomWeight}
                              
                if elmType in {'c', 'rf-c', 'cb', 'rf-c-b'}:
                    data_dict['bias'] = self.elmSet[i][j].b 
                
                with open(FileName, 'wb') as f:
                    pickle.dump(data_dict , f)
        
    def load(self, path):
        
        for i in range(self.layerNum):
            for j in range(self.elmNums[i]):
                elmType = self.argDictList[i][j].keys()[0]
                FileName = path + "\\" + elmType + "-" + str(i) + "-" + str(j)
            
                with open(FileName, 'rb') as f:
                    data_dict = pickle.load(f)            

                self.elmSet[i][j].LinearWeight = data_dict['LinearWeight']
                self.elmSet[i][j].RandomWeight = data_dict['RandomWeight'] 
                
                if elmType in {'c', 'rf-c', 'cb', 'rf-c-b'}:
                    self.elmSet[i][j].b = data_dict['bias']                 