# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:54:28 2016

@author: zz
"""

from numpy import zeros, array, concatenate, random
import cv2
import os

def loadData(folder, gray=True, img_size = [28,28], hog=False, trainVStest = 0.8, percent = 0.1):
    print ">>> Loading Synthetic Data<<<"    
    
    train_data = []
    train_l = []
    test_data = []
    test_l = []

    for i in range(10):
        path = r'D:\workspace\Data\Data Synthesis\\'+folder+'\\'+str(i)+'\\'
        dirs = os.listdir(path)
        for item in dirs: 
            img = cv2.imread(path+item) 
            img = cv2.resize(img, (img_size[1], img_size[0]), interpolation = cv2.INTER_LINEAR)
            if gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if hog:
                pass
            trainOrtest = random.binomial(1, trainVStest)
            if trainOrtest:
                train_data.append(img.flatten())
                train_l.append(i)
            else:
                test_data.append(img.flatten())
                test_l.append(i)
        print 10*(i+1), "% data loaded ... "

    train_data = array(train_data)
    N, M = train_data.shape
    train_label = zeros((N, 10))
    train_l = array(train_l)
    train_label[range(N),train_l] = 1
    data = concatenate((train_data, train_label), axis=1)
    random.shuffle(data)
    needed = int(N*percent)
    train_data = data[:needed,0:M]
    train_label = data[:needed,M:]
    
    test_data = array(test_data)
    N, M = test_data.shape
    test_label = zeros((N, 10))
    test_l = array(test_l)
    test_label[range(N),test_l] = 1
    data = concatenate((test_data, test_label), axis=1)
    random.shuffle(data)
    needed = int(N*percent)
    test_data = data[:needed,0:M]
    test_label = data[:needed,M:]
    
    print ">>> Loading data successful"
    print "Training data shape: ", train_data.shape
    print "Testing data shape: ", test_data.shape
    
    return train_data , train_label, test_data, test_label