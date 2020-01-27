# encoding: utf-8
#有上面這行才能用中文註解
#這個城市只能在Delphes的資料夾下使用
"""
-----------------------------------------------------------
\033[3;32m 你可以打任何你想要的字進來，最好是英文啦 這份是要把“最原始.root”檔做一點分析後轉成一個“.csv”檔和“新的.root” \033[0;m
-----------------------------------------------------------
"""
print(__doc__)  #產生上面那一橫字而已～～
import sys  #載入所需要用的的package
import numpy as np
import ROOT as RT
import time
import pandas as pd

# time counter
t = RT.TStopwatch()
t.Start()
print( time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
ticks_1 = time.time()

#如何使用這個script
if len(sys.argv) < 2:
  print (" Usage:python ppggjet.py ppggjet.root 0r python ../script/ppggqqjet\ -jet\ only.py ppggjet.root")
  sys.exit(1)

#import Delphes library
RT.gSystem.Load("libDelphes")
try:
  RT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
  RT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
except:
    pass

#show program status進度條
def status_line(counter,totalnumber):
    frac = float(counter+1)/float(totalnumber)*100.
    sys.stdout.write('\r')
    #sys.stdout.write("[%-50s] %.2f %%" % ('='*int(nr/2.), nr))
    sys.stdout.write("[%-50s] %i / %i " % ('='*int(frac/2.), (counter+1),int(totalnumber)))
    sys.stdout.flush()

#主程式，載入資料及分析
#最後目標是會在root檔裡建立不同的tree，之後再另外寫程式去畫圖
#main program
def run_data(input,process):
#================ Extracting Data from Delphes root =====================#
    inputFile = input
    chain = RT.TChain("Delphes")
    chain.Add(inputFile)
    # Create object of class ExRootTreeReader
    treeReader = RT.ExRootTreeReader(chain)
    numberOfEntries = treeReader.GetEntries() #他會計算所有的event數量
    tree = RT.TTree(process,process);
    # Get pointers to branches used in this analysis
    """
    下面這幾行是要載入root裡面的branch，有哪些branch可以去Delphes網站上查詢，以下只有載入Jet、Tower(記載jet constituents的).....ㄋ
    """
    branchJet = treeReader.UseBranch("Jet")
    branchTower = treeReader.UseBranch("Tower")
    branchElectron = treeReader.UseBranch("Electron")
    branchMuon = treeReader.UseBranch("Muon")
    branchMissingET = treeReader.UseBranch("MissingET")
    #give variable address
    jpt0,ETA0,PHI0,E,jpt,ETA,PHI,r,JET = [0],[0],[0],[0],[0],[0],[0],[0],[[],[],[],[],[],[],[],[],[],[]]
    list = [["E",np.array(E)],["jpt",np.array(jpt)],["ETA",np.array(ETA)],["PHI",np.array(PHI)],["r",np.array(r)]]
    list2 = ["E","jpt","ETA","PHI","ETA0","PHI0","jpt0","M","M0","jn"]
    j=0
    jn=-1
    var=[]
    var2=[]
    #book branch
    for element in list:
       tree.Branch(element[0],element[1],element[0]+"/D")
    #book pandas for csv
    to,tr,ge,hi=0,0,0,0
    df_2 = pd.DataFrame()
    for entry in range(numberOfEntries): #要loop所有的event
  
      status_line(entry,numberOfEntries)
      # Load selected branches with data from specified event
      treeReader.ReadEntry(entry)
      # there are four leptons at least
      njet = branchJet.GetEntries() #讀取這個event中的jet的數量
      if njet == 0:
            continue
      for i in range(njet):
        jet=branchJet.At(i)
        jetP4=jet.P4()
        ETA0=jetP4.Eta()
        PHI0=jetP4.Phi()
        jpt0=jet.PT
        M0=jet.Mass
        jn=jn+1

        for k in range(jet.Constituents.GetEntriesFast()):
            hi +=1
            subjet = jet.Constituents.At(k)
            if subjet == None:
                continue
            elif subjet.IsA() == RT.Tower.Class():
                tower = RT.Tower(subjet)
                to +=1
            elif subjet.IsA() == RT.Track.Class():
                tower = RT.Track(subjet)
                tr +=1
            elif subjet.IsA() == RT.GenParticle.Class():
                tower = RT.GenParticle(subjet)
                ge +=1
            momentum1 = tower.P4()
            ETA=(momentum1.Eta())
            jpt=(momentum1.Pt())
            E=(momentum1.E())
            PHI=(momentum1.Phi())
            M=(momentum1.M())
            JET[0].append(E)
            JET[1].append(jpt)
            JET[2].append(ETA)
            JET[3].append(PHI)
            JET[4].append(ETA0)
            JET[5].append(PHI0)
            JET[6].append(jpt0)
            JET[7].append(M)
            JET[8].append(M0)
            JET[9].append(jn)
            list = [["E",np.array(E)],["jpt",np.array(jpt)],["ETA",np.array(ETA)],["PHI",np.array(PHI)],["ETA0",np.array(ETA0)],["PHI0",np.array(PHI0)],["jpt0",np.array(jpt0)],["M",np.array(M)],["M0",np.array(M0)]]
            j=j+1

    C={'E':JET[0],'jpt':JET[1],'ETA':JET[2],'PHI':JET[3],'ETA0':JET[4],'PHI0':JET[5],'jpt0':JET[6],'M':JET[7],'M0':JET[8],'jn':JET[9]}    
#          tree.Fill()#把分析填入這個tree中
    df = pd.DataFrame(C,columns=list2) #把var這個list寫進變成pandas的格式，這裡只是暫時的
    df_2 = df_2.append(df,ignore_index=True) #把上面的資料存到之後會存出的pandas 格式裡
       
    df_2.to_csv("/home/james/Documents/"+process+".csv",index = 0)
#     tree.Write()

outputFile = sys.argv[1]  #會抓取你輸入文字的第二項
f=RT.TFile(outputFile,"recreate") #建立一個自己的root檔存放自己分析好的event
run_data("/home/james/Documents/MG5_aMC_v2_6_5/ppgg/Events/run_01/tag_1_delphes_events.root","ppggjet")
run_data("/home/james/Documents/MG5_aMC_v2_6_5/ppqq/Events/run_01/tag_1_delphes_events.root","ppqqjet")
f.Close()

"""
假如要把兩個csv合併成單一個就需要下面這幾段程式
把a.csv and b.csv合併成c.csv
"""
"""
a = pd.read_csv("a.csv")
b = pd.read_csv("b.csv")
out = pd.concat([a,b],ignore_index=True,axis=0,join='inner')
out.to_csv("/c.csv",index = 0)
"""
ticks_2 = time.time()
totaltime =  ticks_2 - ticks_1
t.Stop();
t.Print();
print( "time comsumption : " , totaltime)

