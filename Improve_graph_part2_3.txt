import sys  #載入所需要用的的package
import numpy as np
#import ROOT as RT
import pandas as pd
import csv
import matplotlib.pyplot as plt
import pickle
##--------------------------------------------------------------------------------------------------read data dij

dfrecjg={'jn':[],'dij':[]}

dfrecjg["jn"]=pd.read_csv('/home/ja2006203966/data2/gluon/data2frecjgjn.csv')
dfrecjg["dij"]=pd.read_csv('/home/ja2006203966/data2/gluon/data2frecjg.csv')
del dfrecjg["dij"]['Unnamed: 0']
del dfrecjg["jn"]['Unnamed: 0']
Gpt0=np.load('/home/ja2006203966/data2/gluon/jpt0.npy',allow_pickle=1)
Gpt0=Gpt0.tolist()
Gpt0=[[j for j in i if j>=0] for i in Gpt0  ]
R2=np.load('/home/ja2006203966/data2/R2.npy')
R2=R2.tolist()
G=[]
N0=max(dfrecjg["jn"]["0"])+1
for k in range(N0):
    if k<=(2*N0/3):
        continue
    if k%1000==0:
        print(k)
    gg=[]
    dfabc=dfrecjg["dij"][dfrecjg["jn"]["0"]==k]
    dfabc=dfabc*Gpt0[k][0]*Gpt0[k][0]
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
##----------------------------------------------------------------------------------------------------------------
np.save('/home/ja2006203966/data2/gluon/imGdcut_'+str(R2)+'_3', G)
np.save('/home/ja2006203966/data2/gluon/imGndcut_'+str(R2)+'_3', Gn)
print('gluon complete')
##----------------------------------------------------------------------------------------------------------------
