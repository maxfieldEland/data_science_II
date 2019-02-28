# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 08:40:44 2019

@author: mgreen13
"""
import numpy as np
import matplotlib.pyplot as plt

def gen1(X):
    Y = []
    for x in X:
        y = np.random.binomial(1,(np.exp(4*x-2)/(1+np.exp(4*x-2))))
        Y.append(y)
    return(Y)
        

def gen2(X,Y):
    Z = []
    for idx,y in enumerate(Y):
        x = X[idx]
        z=np.random.binomial(1,np.exp(2*(x+y)-2)/(1+np.exp(2*(x+y)-2)))
        Z.append(z)
    return(Z)

def simulate(n):
    X = np.random.binomial(1,.5, n)
    Y = 1
    Z = np.array(gen2(X,Y))
    # find p( Z = 1 | Y = 1)
    z_1 = Z == 1
    
    p_z_y = 0
    for i in range(n):
        if y_1[i] == z_1[i]:
            p_z_y += 1
            
   
    return(p_z_y/n)
    
def simulate_int(n):
    X = np.random.binomial(1,.5, n)
    Y = np.ones(n)
    Z = np.array(gen2(X,Y))
    # find p( Z = 1 | Y = 1)
    p_z_y = sum(Z)
    
    return(p_z_y/n)
    
p = []
ns = list(range(1,1000))
for i in range(1,1000):
    p.append(simulate_int(i))
    
plt.title("P(Z = 1|Y := 1) over n trials")
plt.xlabel("Trial Length (n)")
plt.ylabel("P(Z = 1 | Y :=1)")
plt.plot(ns,p)