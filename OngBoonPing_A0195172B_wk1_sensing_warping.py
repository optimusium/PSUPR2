# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 09:05:52 2019

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
l2D  = pd.read_csv('ecg2D.csv',                       header=None) 
ecg2D= l2D[1].values

sg  = np.array(ecg2D)

from scipy.signal import find_peaks as findPeaks 
(Pks,_) = findPeaks(sg,height=2) 

#print(len(ecg2D) )
#print(Pks)
def extractECG(ecg,pks,offset=15):
    ecgarr=np.split(ecg, pks-offset)
    return ecgarr


ecgarr=extractECG(ecg2D,Pks,15)
'''
from scipy import sparse 
from scipy.sparse.linalg import spsolve 
def alsbase(y, lam, p, niter=10):
    L = len(y)       
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))       
    w = np.ones(L)            
    for i in range(niter):           
        W = sparse.spdiags(w, 0, L, L)           
        Z = W + lam * D.dot(D.transpose())           
        z = spsolve(Z, w*y)           
        w = p * (y > z) + (1-p) * (y < z)       
    return z
'''
plt.style.use('ggplot')  
plt.rcParams['ytick.right']     = True 
plt.rcParams['ytick.labelright']= True 
plt.rcParams['ytick.left']      = False 
plt.rcParams['ytick.labelleft'] = False

def getDist(x,y):
    dists = np.zeros((len(y),len(x))) 
    for i in range(len(y)):       
        for j in range(len(x)):           
            dists[i,j]  = (y[i]-x[j])**2
    return dists

def getAcuCost(dists):
    acuCost     = np.zeros(dists.shape) 
    acuCost[0,0]= dists[0,0] 
    for j in range(1,dists.shape[1]):        
        acuCost[0,j]    = dists[0,j]+acuCost[0,j-1] 
    for i in range(1,dists.shape[0]):        
        acuCost[i,0]    = dists[i,0]+acuCost[i-1,0]   
    for i in range(1,dists.shape[0]):       
        for j in range(1,dists.shape[1]):           
            acuCost[i,j]    = min(acuCost[i-1,j-1], acuCost[i-1,j],  acuCost[i,j-1])+dists[i,j] 
    #print(acuCost)
    return acuCost


def pltDistances(dists,xlab="X",ylab="Y",clrmap="viridis"):       
     imgplt  = plt.figure()       
     plt.imshow(dists,                  interpolation='nearest',                  cmap=clrmap)            
     plt.gca().invert_yaxis()       
     plt.xlabel(xlab)       
     plt.ylabel(ylab)       
     plt.grid()       
     plt.colorbar()            
     return imgplt  

def getPath(x,y,acuCost):
    i       = len(y)-1 
    j       = len(x)-1                                            
    path    = [[j,i]]                          
    while (i > 0) and (j > 0):       
        if i==0:           
            j   = j-1                                  
        elif j==0:           
            i   = i-1                                  
        else:                                                       
            if acuCost[i-1,j] == min(acuCost[i-1,j-1], acuCost[i-1,j], acuCost[i,j-1]):               
                i   = i-1              
            elif acuCost[i,j-1] == min(acuCost[i-1,j-1], acuCost[i-1,j], acuCost[i,j-1]):               
                j   = j-1           
            else:               
                i   = i-1               
                j   = j-1                   
        path.append([j,i]) 
    path.append([0,0])
    return path
   
def pltCostAndPath(acuCost,path,xlab="X",ylab="Y",clrmap="viridis"):     
    px      = [pt[0] for pt in path]     
    py      = [pt[1] for pt in path]         
    imgplt  = pltDistances(acuCost, xlab=xlab, ylab=ylab, clrmap=clrmap)       
    plt.plot(px,py)          
    return imgplt 

def get_path_cost(path,dists):
    cost        = 0 
    for [j,i] in path:       
        cost    = cost+dists[i,j] 
    print(cost)
    return cost

def pltWarp(s1,s2,path,xlab="idx",ylab="Value"):     
    imgplt      = plt.figure()          
    for [idx1,idx2] in path:         
        plt.plot([idx1,idx2],[s1[idx1],s2[idx2]],   color="C4", linewidth=2)     
        plt.plot(s1,              'o-',              color="C0",              markersize=3)     
        plt.plot(s2,              's-',              color="C1",              markersize=2)     
        plt.xlabel(xlab)  
        plt.ylabel(ylab)          
    return imgplt 

def warpTarget(x,y):
    print("plotting segment %s vs segment %s" % (x,y))
    dists=getDist( ecgarr[x],ecgarr[y] )
    acuCost=getAcuCost(dists) 
    imgplt=pltDistances(acuCost,clrmap='Reds') 
    imgplt.show()
    path=getPath(ecgarr[x],ecgarr[y],acuCost)
    cost=get_path_cost(path,dists)
    imgplt=pltCostAndPath(acuCost,path,clrmap='Reds')
    imgplt.show()
    imgplt=pltWarp(ecgarr[x],ecgarr[y],path)
    imgplt.show()
    #time.sleep(10)

warpTarget(1,2)
warpTarget(2,3)
warpTarget(3,6)
    
'''
plt.figure() 
plt.plot(x,            color="C0",            label='x') 
plt.plot(y,            color="C1",            label='y') 
plt.legend()
'''

'''    



ecgbase     = alsbase(ecg2D, 10^5,0.000005,niter=50) 
ecgcorr     = ecg2D-ecgbase

plt.figure() 
plt.subplot(211) 
plt.plot(ecg2D) 
plt.plot(ecgbase,             color="C1",            linestyle='dotted') 
plt.subplot(212) 
plt.plot(ecgcorr) 
'''
