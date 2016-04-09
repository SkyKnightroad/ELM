# -*- coding: utf-8 -*-
"""
Greville and OPIUM method for classifying MNIST from:
J. Tapson and A. van Schaik, 
"Learning the Pseudoinverse Solution to Network Weights"
Neural Networks

Used for Figure 2.

@author: andrevanschaik
"""

from numpy import vectorize,zeros, ones,insert,random, eye, tanh, dot, reshape, arange, savetxt, loadtxt
from OPIUM import *
from mnist import read
from bitstring import *
import time
import subprocess
import os

start_time = time.time()
#Note: mnist is used to read the database. It expects the datasets to be in a
#directory called 'data' which is a subdirectory of the directory containing the code.
#The datafiles should be called:
#    datasets = { "test"    :   { "images" : "t10k-images-idx3-ubyte", "labels" : "t10k-labels-idx1-ubyte" },
#                 "train"     :   { "images" : "train-images-idx3-ubyte", "labels" : "train-labels-idx1-ubyte" }};

N_A = 64       # number of neurons in first population
N_samples = 256  # number of sample points to use when finding decoders
N_units = 256 
# Network parameters
size_input=28*28
size_hidden=N_A*N_units   # size of hidden layer
W_A = 128*6
M=zeros((10,size_hidden))  # initial value of linear weights
N_train = 60000
N_test = 10000

def LFSR_20bit (lfsr_i):
    lfsr_o = zeros(20,'int')
    for x in range (0,20):
        if ( x == 0 ):
            lfsr_o[0] = int(lfsr_i[19])
        elif ( x == 3 ):
            lfsr_o[3] = int(lfsr_i[x-1])^int(lfsr_i[19])
        else:
            lfsr_o[x] = int(lfsr_i[x-1])
        return lfsr_o

def dec2bin (N_bits,l_index):
    lfsr_bin = zeros(N_bits,'int')
    l_index_div = l_index
    for x in range (0,N_bits):
        l_index_remainder = l_index_div%2
        l_index_div = l_index_div/2
        
        #print x,l_index_div,l_index_remainder
        lfsr_bin[x] = int(l_index_remainder)
	return lfsr_bin

def bin2dec (N_bits,lfsr_bin):
    lfsr_dec = 0
    for x in range (0,N_bits):
        #print x,l_index_div,l_index_remainder
        lfsr_dec = lfsr_dec + lfsr_bin[x]*(2**x) 
    return lfsr_dec

def random_weights_gen (seed):
    #random.seed(seed)
    random_weights = zeros((size_hidden,size_input))
    lfsr_col_i = zeros((size_input/(4*4),20))
    #lfsr_out = open(lfsr_outputFile, "w")
    for i in range (0,size_input/(4*4)-1):
        #rand_id = int(random.uniform(0,2**20))
        #lfsr_col_i[i] = dec2bin(20,rand_id)
        lfsr_col_i[i] = dec2bin(20,i*i*i+seed)

    lfsr_col_o = zeros(20,'int')
    lfsr_o_0 = zeros(5,'int')
    lfsr_o_1 = zeros(5,'int')
    lfsr_o_2 = zeros(5,'int')
    lfsr_o_3 = zeros(5,'int')

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
                
                #if ( step == 3 and col == size_input/(4*4) - 1 ):
                #    lfsr_out.write("%s,%s,%s,%s" % (col_int_0,col_int_1,col_int_2,col_int_3))
                #else:
                #    lfsr_out.write("%s,%s,%s,%s," % (col_int_0,col_int_1,col_int_2,col_int_3))
                lfsr_col_i[col] = lfsr_col_o

    #lfsr_out.write("\n")
    #lfsr_out.close()
    return random_weights

# Initialisation of signal matrices
def read_tuning_curves():
    
    x_values=arange(-1.0,1.0,2.0/N_samples)
    A = zeros((N_samples, N_A))
    
    for row in range (0,N_samples):
        for column in range (0,N_A/2):
            col = column
            y = int(col*(-row+N_samples-1-(col*4)))/32
            if (y > 0 ):
                A[row,column] = int((y))
            else:
                A[row,column] = 0
                for column in range (N_A/2,N_A):
                    col = column - (N_A/2 - 1)
                    y = int(col*(row-(col*4)))/32
                    if (  y > 0 ):
                        A[row,column] =  int((y))
                    else:
                        A[row,column] = 0
    return x_values, A

x,A_64 = read_tuning_curves()
A = zeros((N_samples,size_hidden))
for f in range (0,N_samples):
    for g in range (0,N_units):
        for j in range (0,N_A): 
            A[f,g*N_A+j] = A_64[f,j]

def M_gen (random_weights):
    dataset = mnist("train")
    h=zeros((size_hidden,1))
    img_int = zeros((size_input,1))
    
    for i in range (0,N_train):
        label, img = dataset.GetImage(i)
        label_vector = zeros((10,1))
        label_vector[label]=1.0
        
        for j in range (0, size_input):
            if ( img[j] > 0 ):
                img_int[j] = float(1.0)
            else:
                img_int[j] = float(0.0)
                
        #h = tanh(dot(random_weights,img_int)/16.0)
        vin_tmp = dot(random_weights,img_int)
        for t in range (0, size_hidden):
            h[t] = A[abs(int(vin_tmp[t]))%256,t%64]/128.0 
            
        y = dot(M,h)
        e = reshape(label_vector,(10,1))-y       
        # Choice between the Greville and OPIUM method    
        #OPIUM(h,e,M,Theta)
        OPIUMl(h,e,M,1)
        
        M_float = M
        M_int=zeros((10,size_hidden))
        for col in range (0,size_hidden):
            for p_index in range (0,10):
                temp = int(M[p_index][col]*W_A)
                M_int[p_index][col] = temp
    return M_float,M_int

# Load the testing dataset
def test_gen(M_int,random_weights,M_float): 
    f_errors = 0
    errors = 0
    dataset = mnist("test")
    
    img_rec = zeros((size_input,1))
    for q in range (0,N_test):
        label, img = dataset.GetImage(q)
        
        for r in range (0,size_input):
            if ( img[r] > 0 ):
                img_rec[r] = float(1.0)
            else:
                img_rec[r] = float(0.0)

        vin_tmp = dot(random_weights,img_rec)
        for t in range (0, size_hidden):
            h[t] = A[abs(int(vin_tmp[t]))%256,t%64]/128.0
            
    #h = tanh(dot(random_weights,img_rec)/16.0)
    y = dot(M_float,h)
    D = y.argmax()
    y_int = dot(M_int,h)
    D_int = y_int.argmax()
    if (label != D):
        f_errors = f_errors + 1   
    if (label != D_int):
        errors = errors + 1
        
    print f_errors
    return errors

min_seed = 0
min_errors = N_test
min_lfsr_col = zeros((size_input/(4*4),20))
run_times = 0
h_train = zeros((N_train,size_hidden))
h_test = zeros((size_hidden,N_test))
h = zeros((size_hidden,1))
seed = 198964

while ( run_times < 100 ):
    #seed = int(random.uniform(0,2**20)) 
    random_weights = random_weights_gen(seed)
    M_float,M_int = M_gen(random_weights)
    errors = test_gen(M_int,random_weights,M_float)
    if ( errors < min_errors ):
        min_errors = errors
        min_seed = seed
        #min_lfsr_col = lfsr_col_i
    run_times = run_times + 1
    print seed,errors
    seed = seed + 1234

end_time = time.time()
print("Elapsed time was %g seconds" % (end_time - start_time))
