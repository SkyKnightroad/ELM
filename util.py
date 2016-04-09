# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:03:47 2016

@author: zz
"""
from MNISTDataset import MNISTDataset
from bitstring import Bits
import numpy as np

def loadData(percent):
    # load data set
    path = 'D:\workspace\Data\ELM\MNIST'
    loader = MNISTDataset(path)

    dim = loader.itemDimension
    feature_dim = dim[0]*dim[1]
    label_dim = loader.labelDimension[0]

    train_size = int(loader.numberOfTrainingItems * percent)
    test_size = int(loader.numberOfTestingItems * percent)
    train_data = np.zeros((train_size, feature_dim))
    train_label = np.zeros((train_size, label_dim))
    test_data = np.zeros((test_size, feature_dim))
    test_label = np.zeros((test_size, label_dim))

    print "Loading data"
    for i in range(train_size):
        train_label[i], train_data[i] = loader.getTrainingItem(i)
    
    for i in range(test_size):
        test_label[i], test_data[i] = loader.getTestingItem(i)
    print "Loading data succeed"
    print "Training data set size: ", train_size, "Testing data set size: ", test_size
    
    return train_data, train_label, test_data, test_label
    
def normalizeData(data):
    data_mean = data.mean(axis = 0)
    data_std = data.std(axis = 0) 
    data_std[data_std == 0] = 1   
    return (data - data_mean)/data_std
    
# simulate using n bits to represente the given decimal number
def around(decimal, N_bits):
    d = np.around(decimal, decimals = 0)
    f = decimal - d
    p = 2**-N_bits
    f = np.around(f/p, decimals = 0)*p
    return d+f
    
def LFSR_20bit (lfsr_i):
    lfsr_o = np.zeros(20,'int')
    
    for x in range (0,20):
        if ( x == 0 ):
            lfsr_o[0] = int(lfsr_i[19])
        elif ( x == 3 ):
            lfsr_o[3] = int(lfsr_i[x-1])^int(lfsr_i[19])
        else:
            lfsr_o[x] = int(lfsr_i[x-1])
    
    return lfsr_o

def dec2bin (N_bits,l_index):
    lfsr_bin = np.zeros(N_bits,'int')
    l_index_div = l_index
    for x in range(N_bits):
        l_index_remainder = l_index_div%2
        l_index_div = l_index_div/2
        lfsr_bin[x] = int(l_index_remainder)
        
    return lfsr_bin

def bin2dec (N_bits,lfsr_bin):
    lfsr_dec = 0
    for x in range (0,N_bits):
        lfsr_dec = lfsr_dec + lfsr_bin[x]*(2**x)
    return lfsr_dec

def random_weights_gen (size_input, size_hidden, seed=235733):
    random_weights = np.zeros((size_hidden,size_input))
    lfsr_col_i = np.zeros((size_input/(4*4),20))
    for i in range (0,size_input/(4*4)-1):
        lfsr_col_i[i] = dec2bin(20,i*i*i+seed)

    lfsr_col_o = np.zeros(20,'int')
    lfsr_o_0 = np.zeros(5,'int')
    lfsr_o_1 = np.zeros(5,'int')
    lfsr_o_2 = np.zeros(5,'int')
    lfsr_o_3 = np.zeros(5,'int')

    for row in range (0,size_hidden):
        for step in range (0,4):
            col_off = 196*step
            for col in range (0,size_input/(4*4)):
                lfsr_col_o = LFSR_20bit(lfsr_col_i[col])
                for path_i in range (0,5):
                    lfsr_o_0[path_i] = lfsr_col_o[4-path_i]
                    lfsr_o_1[path_i] = lfsr_col_o[4-path_i+5]
                    lfsr_o_2[path_i] = lfsr_col_o[4-path_i+10]
                    lfsr_o_3[path_i] = lfsr_col_o[4-path_i+15] 
                    
                col_int_0 = Bits(lfsr_o_0).int
                col_int_1 = Bits(lfsr_o_1).int
                col_int_2 = Bits(lfsr_o_2).int
                col_int_3 = Bits(lfsr_o_3).int
                
                random_weights[row,col_off+col*4+0] = col_int_0
                random_weights[row,col_off+col*4+1] = col_int_1
                random_weights[row,col_off+col*4+2] = col_int_2
                random_weights[row,col_off+col*4+3] = col_int_3
                
                lfsr_col_i[col] = lfsr_col_o

    return random_weights