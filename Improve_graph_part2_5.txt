import sys  #載入所需要用的的package
import numpy as np
#import ROOT as RT
import pandas as pd
import csv
import matplotlib.pyplot as plt
import pickle
##--------------------------------------------------------------------------------------------------read data dij

dfrecjq={'jn':[],'dij':[]}
dfrecjq["jn"]=pd.read_csv('/home/ja2006203966/data2/quark/data2frecjqjn.csv')
dfrecjq["dij"]=pd.read_csv('/home/ja2006203966/data2/quark/data2frecjq.csv')
del dfrecjq["dij"]['Unnamed: 0']
del dfrecjq["jn"]['Unnamed: 0']
##----------------------------------------------------------------------------------------------------------------
R2=np.load('/home/ja2006203966/data2/R2.npy')
R2=R2.tolist()
G2pt0=np.load('/home/ja2006203966/data2/quark/G2jpt0.npy',allow_pickle=1)
G2pt0=G2pt0.tolist()
G=[]
N0=max(dfrecjq["jn"]["0"])+1
for k in range(N0):
    if k<=(N0/3):
        continue
    if k>(2*N0/3):
        break
        
    if k%1000==0:
        print(k)
    gg=[]
    dfabc=dfrecjq["dij"][dfrecjq["jn"]["0"]==k]
    dfabc=dfabc*G2pt0[k]*G2pt0[k]
    dfabc=dfabc[dfabc<R2].fillna(0)
    N=dfabc.shape[0]
    N2=dfabc.shape[1]
    if N<=N2:
        N2=N
    dfabc.index = range(len(dfabc))
    dfabc=dfabc.T
    for j in range(N):
        test=[i for i in range(j) if (dfabc[j][i]!=0)&(i<=(j-1))]+[i+1 for i in range(N2) if (dfabc[j][i]!=0)&(i>(j-1))]
        gg.append(test)
    G.append(gg)

Gn=[]

for i in G:
    gg=[]
    for j in i:
        gg.append(set(j))
    Gn.append(gg)

np.save('/home/ja2006203966/data2/quark/imG2dcut_'+str(R2)+'_2', G)
np.save('/home/ja2006203966/data2/quark/imGn2dcut_'+str(R2)+'_2', Gn)