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
dta2.index = pd.Index(sm.tsa.datetools.dates_from_range('1000',length=len(dta2)))
#dta2.index = pd.Index(range(0, len(dta2)))
dta2.plot(figsize=(12,8))


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


#arma_mod20 = sm.tsa.ARMA(dta3,(7,0)).fit()
#print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
#arma_mod30 = sm.tsa.ARMA(dta3,(0,1)).fit()
#print(arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)
#arma_mod40 = sm.tsa.ARMA(dta3,(7,1)).fit()
#print(arma_mod40.aic,arma_mod40.bic,arma_mod40.hqic)
#arma_mod50 = sm.tsa.ARMA(dta3,(8,0)).fit()
#print(arma_mod50.aic,arma_mod50.bic,arma_mod50.hqic)

