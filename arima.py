# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:14:15 2018

@author: eqhuliu
"""

from __future__ import print_function

import numpy as np

import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from scipy import  stats
import matplotlib.pyplot as plt



#dta=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422, 
#6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355, 
#10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767, 
#12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232, 
#13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248, 
#9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722, 
#11999,9390,13481,14795,15845,15271,14686,11054,10395]

dta2=pd.read_csv('delay_data_test.csv',header=None)
#dta2.index = pd.Index(sm.tsa.datetools.dates_from_range('1000m1',length=len(dta2)))
dta2.index = pd.to_datetime(dta2.index, unit='ms')
#dta2.index = pd.Index(range(0, len(dta2)))
dta2.plot(figsize=(12,8))



dta3=pd.read_csv('delay_data.csv',header=None)
#dta2.index = pd.Index(sm.tsa.datetools.dates_from_range('1000m1',length=len(dta2)))
pd.to_datetime(dta3.index, unit='ms')
dta3.index = pd.to_datetime(dta3.index, unit='ms')
#dta2.index = pd.Index(range(0, len(dta2)))

#dta=pd.Series(dta)
#dta.index = pd.Index(sm.tsa.datetools.dates_from_range('2001','2090'))
#dta.plot(figsize=(12,8))


#dta.tail()

fig = plt.figure(figsize=(12,8))

ax1= fig.add_subplot(121)
diff1 = dta2.diff(1)
diff1.plot(ax=ax1)

ax2= fig.add_subplot(122)
diff2 = dta2.diff(2)
diff2.plot(ax=ax2)


diff1= dta2.diff(1)

fig = plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta2,lags=40,ax=ax1)

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta2,lags=40,ax=ax2)




arma_mod21 = sm.tsa.ARMA(dta2.astype('float64'),(2,1)).fit()
arma_mod20 = sm.tsa.ARMA(dta2.astype('float64'),(2,0)).fit()
print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
arma_mod01 = sm.tsa.ARMA(dta2.astype('float64'),(0,1)).fit()
print(arma_mod01.aic,arma_mod01.bic,arma_mod01.hqic)
print(arma_mod21.aic,arma_mod21.bic,arma_mod21.hqic)
arma_mod30 = sm.tsa.ARMA(dta2.astype('float64'),(3,0)).fit()
print(arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)
arma_mod31 = sm.tsa.ARMA(dta2.astype('float64'),(3,1)).fit()
print(arma_mod31.aic,arma_mod31.bic,arma_mod31.hqic)


#mod21-->#173.32772855124503 186.738384687 178.765262518
#174.55178448720673 185.280309396 178.901811661
#206.098776562232 214.145170244 209.361296942
#174.03232305944348 187.442979195 179.469857026
#175.22216116088097 191.314948524 181.747201921

#so the best is mod21

resid = arma_mod21.resid 

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)

print(sm.stats.durbin_watson(arma_mod21.resid.values))
#1.94381530456

print(stats.normaltest(resid))
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)

#Q/Jung-Box
#r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
#data = np.c_[range(1,41), r[1:], q, p]
#table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
#print(table.set_index('lag'))

#predict

predict_dta = arma_mod21.predict(108, 215, dynamic=True)
#print(predict_dta)
#print(dta2)

fig = plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
dta4=dta3[108:216]
dta4.plot(ax=ax1)
#ax2 = fig.add_subplot(312)
#predict_dta.plot(ax=ax2)


ax3 = fig.add_subplot(212)


diff_shift_ts = diff1.shift(1)
diff_shift_ts.index = pd.to_datetime(predict_dta.index, unit='ms')
diff_recover = predict_dta.add(diff_shift_ts[0])
diff_recover.dropna(inplace=False)
print(diff_recover)
diff_recover.plot(ax=ax3)


fig, ax = plt.subplots(figsize=(12, 8))
ax = dta2.loc[dta2.index[0:]].plot(ax=ax)
fig = arma_mod21.plot_predict(108, 215, dynamic=True, ax=ax, plot_insample=False)
plt.show()

dta3.plot(figsize=(12,8))

def getReward(actual,prediction,s):
    if(np.abs(actual - prediction) <=s):
        return 1
    else:
        return 0

sum=0
for i in range(108,215):
    sum=sum+getReward(dta3[0][i],diff_recover[i-108],2)
print("Limition range =2,The rewards is:%d"%sum)
print("The result is:%f"%(sum/(215-108)))

sum=0
for i in range(108,215):
    sum=sum+getReward(dta3[0][i],diff_recover[i-108],1)
print("Limition range =1,The rewards is:%d"%sum)
print("The result is:%f"%(sum/(215-108)))
