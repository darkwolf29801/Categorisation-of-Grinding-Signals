import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
import matplotlib.pyplot as mt
from scipy.fft import fft

folder=os.listdir('CSV')

file=pd.read_csv('D:/PALS/CSV/6.csv')
#mt.plot(file['Power'])
fig=mt.figure()

#mt.plot(file['Untitled 1'])
disp=np.array(file['Untitled 1'])

file=np.array(file['Power'])

fig1=mt.figure()

'''
mt.plot(fft(file))
'''
def noise_cancel(file):
    res=[]
    for i in range(0,len(file),10):
        res.append(sum(file[i:i+10])/2)
    return res


#mt.plot(noise_cancel(disp))

j=disp[0]
part,part2=[],[]
for i in range(len(disp)):
    if disp[i]==j:
        part.append(file[i])
    else:
        break
        part2.append(file[i])
        part.append(0)

mt.plot(part,"r")
mt.plot(part2,"b")
