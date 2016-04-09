# -*- coding: utf-8 -*-
from numpy import dot, exp, eye, sqrt

def OPIUMl(act, err, w, theta, alpha): #OPIUM light
    norm = (1.0/theta) + dot(act.T,act)   
    w += dot(err,act.T)/norm
    
    return w, theta

def OPIUM(act, err, w, theta, alpha): #basic OPIUM
    act_spred = dot(theta,act)                      
    norm = 1+dot(act.T,act_spred)                
    theta -= dot(act_spred,act_spred.T)/norm
    w += dot(err,act_spred.T)/norm 
    
    return w, theta

"""
def OPIUMd(act, err, w, theta, alpha): #dynamic OPIUM 
    act_spred = dot(theta,act)                      
    norm1 = 1+dot(act.T,act_spred)   
    norm2 = 1+alpha*(1-exp(-sqrt(dot(err.T,err))/err.size))
    theta -= dot(act_spred,act_spred.T)/norm1
    theta += alpha * eye(theta.size**0.5) * (1-exp(-sqrt(dot(err.T,err))/err.size))
    theta /= norm2
    w += dot(err,act_spred.T)/norm1 
"""
    
def OPIUMd(act, err, w, theta, alpha): #dynamic OPIUM 
    act_spred = dot(theta,act)                      
    norm1 = 1+dot(act.T,act_spred)
    err_L2 = sqrt(dot(err.T,err))
    M = err.size
    E = err_L2/M/norm1
    norm2 = 1+alpha*(1-exp(-E))
    theta -= dot(act_spred,act_spred.T)/norm1
    theta +=  alpha * eye(theta.size**0.5) * (1-exp(-E))
    theta /= norm2
    w += dot(err,act_spred.T)/norm1
    
    return w, theta