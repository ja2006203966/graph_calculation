import sys  #載入所需要用的的package
import numpy as np
#import ROOT as RT
import pandas as pd
import csv
import matplotlib.pyplot as plt
import pickle
import os


R2=9
np.save('/home/ja2006203966/data2/R2',R2)


# os.system("nohup python -u /home/ja2006203966/script/graph3/Improve_graph_part2_1.py>imd_g1 2>&1 &")
# os.system("nohup python -u /home/ja2006203966/script/graph3/Improve_graph_part2_2.py>imd_g2 2>&1 &")
# os.system("nohup python -u /home/ja2006203966/script/graph3/Improve_graph_part2_3.py>imd_g3 2>&1 &")
os.system("nohup python -u /home/ja2006203966/script/graph3/Improve_graph_part2_4.py>imd_q1 2>&1 &")
os.system("nohup python -u /home/ja2006203966/script/graph3/Improve_graph_part2_5.py>imd_q2 2>&1 &")
os.system("nohup python -u /home/ja2006203966/script/graph3/Improve_graph_part2_6.py>imd_q3 2>&1 &")
