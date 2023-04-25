#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:13:56 2021

@author: istirodiah
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

from scipy.optimize import Bounds
bounds = Bounds(np.zeros((24)), 1*np.ones((24)))

def model(opparams, init_vals, period, params, t):
    Sa_0, Sb_0, Sc_0, Sd_0, Ba_0, Bb_0, Vc_0, Vd_0, Eax_0, Eay_0, Ebx_0, Eby_0, Ecx_0, Ecy_0, Edx_0, Edy_0, Iax_0, Iay_0, Ibx_0, Iby_0, Icx_0, Icy_0, Idx_0, Idy_0, Hax_0, Hay_0, Hbx_0, Hby_0, Hcx_0, Hcy_0, Hdx_0, Hdy_0, Uax_0, Uay_0, Ubx_0, Uby_0, Ucx_0, Ucy_0, Udx_0, Udy_0, R_0, D_0, Ns_0, Nh_0, Nu_0 = init_vals
    Sa, Sb, Sc, Sd, Ba, Bb, Vc, Vd, Eax, Eay, Ebx, Eby, Ecx, Ecy, Edx, Edy, Iax, Iay, Ibx, Iby, Icx, Icy, Idx, Idy, Hax, Hay, Hbx, Hby, Hcx, Hcy, Hdx, Hdy, Uax, Uay, Ubx, Uby, Ucx, Ucy, Udx, Udy, R, D, Ns, Nh, Nu = [Sa_0], [Sb_0], [Sc_0], [Sd_0], [Ba_0], [Bb_0], [Vc_0], [Vd_0], [Eax_0], [Eay_0], [Ebx_0], [Eby_0], [Ecx_0], [Ecy_0], [Edx_0], [Edy_0], [Iax_0], [Iay_0], [Ibx_0], [Iby_0], [Icx_0], [Icy_0], [Idx_0], [Idy_0], [Hax_0], [Hay_0], [Hbx_0], [Hby_0], [Hcx_0], [Hdx_0], [Hdy_0], [Hcy_0], [Uax_0], [Uay_0], [Ubx_0], [Uby_0], [Ucx_0], [Ucy_0], [Udx_0], [Udy_0], [R_0], [D_0], [Ns_0], [Nh_0], [Nu_0]
    P1, P3, P4, P5, P9 = period
    c, alphaa, alphaax, betaa, betaax, epsilona, deltaa, deltaax, thetaa, thetaax , alphab, alphabx, betab, betabx, epsilonb, deltab, deltabx, thetab, thetabx, alphac, alphacx, betac, betacx, epsilonc, deltac, deltacx, thetac, thetacx, alphad, alphadx, betad, betadx, epsilond, deltad, deltadx, thetad, thetadx = params
    
    
    dt = (t[1] - t[0])*1./7
    
    
    for i in t[1:]:
        next_Sa  = Sa[-1] - (c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betaa*(1-epsilona)*Sa[-1] + epsilona*P9*Sa[-1])*dt
        next_Ba  = Ba[-1] - (c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betaax*Ba[-1] - epsilona*P9*Sa[-1])*dt
        next_Eax = Eax[-1] + (c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betaax*Ba[-1] - P1*Eax[-1])*dt
        next_Eay = Eay[-1] + (c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betaa*(1-epsilona)*Sa[-1] - P1*Eay[-1])*dt
        next_Iax = Iax[-1] + (P1*Eax[-1] - P3*Iax[-1])*dt
        next_Iay = Iay[-1] + (P1*Eay[-1] - P3*Iay[-1])*dt
        next_Hax = Hax[-1] + (alphaax*P3*Iax[-1] - P4*Hax[-1])*dt
        next_Hay = Hay[-1] + (alphaa*P3*Iay[-1] - P4*Hay[-1])*dt
        next_Uax = Uax[-1] + (deltaax*P4*Hax[-1] - P5*Uax[-1])*dt
        next_Uay = Uay[-1] + (deltaa*P4*Hay[-1] - P5*Uay[-1])*dt
        
        next_Sb  = Sb[-1] - (c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betab*(1-epsilonb)*Sb[-1] + epsilonb*P9*Sb[-1])*dt
        next_Bb  = Bb[-1] - (c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betabx*Bb[-1] - epsilonb*P9*Sb[-1])*dt
        next_Ebx = Ebx[-1] + (c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betabx*Bb[-1] - P1*Ebx[-1])*dt
        next_Eby = Eby[-1] + (c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betab*(1-epsilonb)*Sb[-1] - P1*Eby[-1])*dt
        next_Ibx = Ibx[-1] + (P1*Ebx[-1] - P3*Ibx[-1])*dt
        next_Iby = Iby[-1] + (P1*Eby[-1] - P3*Iby[-1])*dt
        next_Hbx = Hbx[-1] + (alphabx*P3*Ibx[-1] - P4*Hbx[-1])*dt
        next_Hby = Hby[-1] + (alphab*P3*Iby[-1] - P4*Hby[-1])*dt
        next_Ubx = Ubx[-1] + (deltabx*P4*Hbx[-1] - P5*Ubx[-1])*dt
        next_Uby = Uby[-1] + (deltab*P4*Hby[-1] - P5*Uby[-1])*dt
        
        next_Sc  = Sc[-1] - (c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betac*(1-epsilonc)*Sc[-1] + epsilonc*P9*Sc[-1])*dt
        next_Vc  = Vc[-1] - (c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betacx*Vc[-1] - epsilonc*P9*Sc[-1])*dt
        next_Ecx = Ecx[-1] + (c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betacx*Vc[-1] - P1*Ecx[-1])*dt
        next_Ecy = Ecy[-1] + (c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betac*(1-epsilonc)*Sc[-1] - P1*Ecy[-1])*dt
        next_Icx = Icx[-1] + (P1*Ecx[-1] - P3*Icx[-1])*dt
        next_Icy = Icy[-1] + (P1*Ecy[-1] - P3*Icy[-1])*dt
        next_Hcx = Hcx[-1] + (alphacx*P3*Icx[-1] - P4*Hcx[-1])*dt
        next_Hcy = Hcy[-1] + (alphac*P3*Icy[-1] - P4*Hcy[-1])*dt
        next_Ucx = Ucx[-1] + (deltacx*P4*Hcx[-1] - P5*Ucx[-1])*dt
        next_Ucy = Ucy[-1] + (deltac*P4*Hcy[-1] - P5*Ucy[-1])*dt
        
        next_Sd  = Sd[-1] - (c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betad*(1-epsilond)*Sd[-1] + epsilona*P9*Sd[-1])*dt
        next_Vd  = Vd[-1] - (c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betadx*Vd[-1] - epsilond*P9*Sd[-1])*dt
        next_Edx = Edx[-1] + (c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betadx*Vd[-1] - P1*Edx[-1])*dt
        next_Edy = Edy[-1] + (c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betad*(1-epsilona)*Sd[-1] - P1*Edy[-1])*dt
        next_Idx = Idx[-1] + (P1*Edx[-1] - P3*Idx[-1])*dt
        next_Idy = Idy[-1] + (P1*Edy[-1] - P3*Idy[-1])*dt
        next_Hdx = Hdx[-1] + (alphadx*P3*Idx[-1] - P4*Hdx[-1])*dt
        next_Hdy = Hdy[-1] + (alphad*P3*Idy[-1] - P4*Hdy[-1])*dt
        next_Udx = Udx[-1] + (deltadx*P4*Hdx[-1] - P5*Udx[-1])*dt
        next_Udy = Udy[-1] + (deltad*P4*Hdy[-1] - P5*Udy[-1])*dt
        
        next_R  = R[-1] + ((1-alphaax)*P3*Iax[-1] + (1-alphaa)*P3*Iay[-1] + (1-deltaax)*P4*Hax[-1] + (1-deltaa)*P4*Hay[-1] + (1-thetaax)*P5*Uax[-1] + (1-thetaa)*P5*Uay[-1] + 
                           (1-alphabx)*P3*Ibx[-1] + (1-alphab)*P3*Iby[-1] + (1-deltabx)*P4*Hbx[-1] + (1-deltab)*P4*Hby[-1] + (1-thetabx)*P5*Ubx[-1] + (1-thetab)*P5*Uby[-1] + 
                           (1-alphacx)*P3*Icx[-1] + (1-alphac)*P3*Icy[-1] + (1-deltacx)*P4*Hcx[-1] + (1-deltac)*P4*Hcy[-1] + (1-thetacx)*P5*Ucx[-1] + (1-thetac)*P5*Ucy[-1] + 
                           (1-alphadx)*P3*Idx[-1] + (1-alphad)*P3*Idy[-1] + (1-deltadx)*P4*Hdx[-1] + (1-deltad)*P4*Hdy[-1] + (1-thetadx)*P5*Udx[-1] + (1-thetad)*P5*Udy[-1])*dt
        next_D  = D[-1] + (thetaax*P5*Uax[-1] + thetaa*P5*Uay[-1] + thetabx*P5*Ubx[-1] + thetab*P5*Uby[-1] + thetacx*P5*Ucx[-1] + thetac*P5*Ucy[-1] + thetadx*P5*Udx[-1] + thetad*P5*Udy[-1])*dt
        
        next_Ns = Ns[-1] + (c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betaa*(1-epsilona)*Sa[-1] + c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betaax*Ba[-1] +
                   c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betab*(1-epsilonb)*Sb[-1] + c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betabx*Bb[-1] +
                   c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betac*(1-epsilonc)*Sc[-1] + c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betacx*Vc[-1] +
                   c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betad*(1-epsilond)*Sd[-1] + c.dot((Iax[-1]+Iay[-1]+Ibx[-1]+Iby[-1]+Icx[-1]+Icy[-1]+Idx[-1]+Idy[-1])*N)*betadx*Vd[-1])*dt
        next_Nu = Nu[-1] + ((deltaax*Hax[-1] + deltaa*Hay[-1] + deltabx*Hbx[-1] + deltab*Hby[-1] + deltacx*Hcx[-1] + deltac*Hcy[-1] + deltadx*Hdx[-1] + deltad*Hdy[-1]) * P4)*dt
        next_Nh = next_Hax + next_Hay + next_Hbx + next_Hby + next_Hcx + next_Hcy + next_Hdx + next_Hdy
        
        
        Sa  = np.vstack((Sa, next_Sa))
        Sb  = np.vstack((Sb, next_Sb))
        Sc  = np.vstack((Sc, next_Sc))
        Sd  = np.vstack((Sd, next_Sd))
        Ba  = np.vstack((Ba, next_Ba))
        Bb  = np.vstack((Bb, next_Bb))
        Vc  = np.vstack((Vc, next_Vc))
        Vd  = np.vstack((Vd, next_Vd))
        Eax = np.vstack((Eax, next_Eax))
        Eay = np.vstack((Eay, next_Eay))
        Ebx = np.vstack((Ebx, next_Ebx))
        Eby = np.vstack((Eby, next_Eby))
        Ecx = np.vstack((Ecx, next_Ecx))
        Ecy = np.vstack((Ecy, next_Ecy))
        Edx = np.vstack((Edx, next_Edx))
        Edy = np.vstack((Edy, next_Edy))
        Iax = np.vstack((Iax, next_Iax))
        Iay = np.vstack((Iay, next_Iay))
        Ibx = np.vstack((Ibx, next_Ibx))
        Iby = np.vstack((Iby, next_Iby))
        Icx = np.vstack((Icx, next_Icx))
        Icy = np.vstack((Icy, next_Icy))
        Idx = np.vstack((Idx, next_Idx))
        Idy = np.vstack((Idy, next_Idy))
        Hax = np.vstack((Hax, next_Hax))
        Hay = np.vstack((Hay, next_Hay))
        Hbx = np.vstack((Hbx, next_Hbx))
        Hby = np.vstack((Hby, next_Hby))
        Hcx = np.vstack((Hcx, next_Hcx))
        Hcy = np.vstack((Hcy, next_Hcy))
        Hdx = np.vstack((Hdx, next_Hdx))
        Hdy = np.vstack((Hdy, next_Hdy))
        Uax = np.vstack((Uax, next_Uax))
        Uay = np.vstack((Uay, next_Uay))
        Ubx = np.vstack((Ubx, next_Ubx))
        Uby = np.vstack((Uby, next_Uby))
        Ucx = np.vstack((Ucx, next_Ucx))
        Ucy = np.vstack((Ucy, next_Ucy))
        Udx = np.vstack((Udx, next_Udx))
        Udy = np.vstack((Udy, next_Udy))
        R  = np.vstack((R, next_R))
        D  = np.vstack((D, next_D))
        
        Ns  = np.vstack((Ns, next_Ns))
        Nh  = np.vstack((Nh, next_Nh))
        Nu  = np.vstack((Nu, next_Nu))
    
    return Sa, Sb, Sc, Sd, Ba, Bb, Vc, Vd, Eax, Eay, Ebx, Eby, Ecx, Ecy, Edx, Edy, Iax, Iay, Ibx, Iby, Icx, Icy, Idx, Idy, Hax, Hay, Hbx, Hby, Hcx, Hcy, Hdx, Hdy, Uax, Uay, Ubx, Uby, Ucx, Ucy, Udx, Udy, R, D, Ns, Nh, Nu



file1 = pd.ExcelFile('Forecast.xlsx')
df5 = file1.parse(0) #Infection
df6 = file1.parse(1) #CumDeath
df7 = file1.parse(2) #Death
df8 = file1.parse(3) #Hospitalization
df9 = file1.parse(4) #ICU

idata  = df5.values[0:149, 1:7]
cddata = df6.values[0:149, 1:7]
ddata  = df7.values[0:149, 1:7]
hdata  = df8.values[0:149, 1:7]
udata  = df9.values[0:149, 1:7]
   

file = pd.ExcelFile('BoostS.xlsx')
df1 = file.parse(0)  #init
df2 = file.parse(1)  #params
df3  = file.parse(4) #coeficient
df4  = file.parse(5) #opp

#### IGRA
#df1 = file.parse(6)   #init
#df2 = file.parse(7)   #params
#df3  = file.parse(10) #coeficient
#df4  = file.parse(11) #opp

sce = 1
coef = df3.values[1:,1+4*(sce-1)]
coef = coef.astype(float)
coeh = df3.values[1:,2+4*(sce-1)]
coeh = coeh.astype(float)
coeu = df3.values[1:,3+4*(sce-1)]
coeu = coeu.astype(float)
coed = df3.values[1:,4+4*(sce-1)]
coed = coed.astype(float)

pop = df1.values[:, 1:7]
par = df2.values[:, 1:7]
pop = pop.astype(float)
par = par.astype(float)

it = 10
t_max = 7*it
times = np.arange(0, t_max+1, 7)

dt = 1
t  = np.linspace(0, t_max, int(t_max/dt)+1)

### Parameters
P1 = 1./3
P3 = 1./5
P4 = 1./4 #7
P5 = 1./7
P9 = 1./14

c   = np.array([[5.090928e-07, 1.016800e-07, 9.904523e-08, 6.037230e-08, 3.330160e-08, 1.418771e-08],
                [1.016800e-07, 4.277460e-07, 7.586532e-08, 7.709942e-08, 3.231887e-08, 1.384713e-08],
                [9.904523e-08, 7.586532e-08, 2.650871e-07, 1.042291e-07, 3.812121e-08, 2.656602e-08],
                [6.037230e-08, 7.709942e-08, 1.042291e-07, 1.308419e-07, 5.894746e-08, 3.369221e-08],
                [3.330160e-08, 3.231887e-08, 3.812121e-08, 5.894746e-08, 1.187878e-07, 6.106433e-08],
                [1.418771e-08, 1.384713e-08, 2.656602e-08, 3.369221e-08, 6.106433e-08, 8.840770e-08]])  

alphaa   = par[0]
# alphaax  = par[1]
betaa    = par[2]
# betaax   = par[3]
epsilona = par[4]
deltaa   = par[5]
# deltaax  = par[6]
thetaa   = par[7]
# thetaax  = par[8]
alphab   = par[9]
# alphabx  = par[10]
betab    = par[11]
# betabx   = par[12]
epsilonb = par[13]
deltab   = par[14]
# deltabx  = par[15]
thetab   = par[16]
# thetabx  = par[17]
alphac   = par[18]
# alphacx  = par[19]
betac    = par[20]
# betacx   = par[21]
epsilonc = par[22]
deltac   = par[23]
# deltacx  = par[24]
thetac   = par[25] 
# thetacx  = par[26]
alphad   = par[27]
# alphadx  = par[28]
betad    = par[29]
# betadx   = par[30]
epsilond = par[31]
deltad   = par[32]
# deltadx  = par[33]
thetad   = par[34]
# thetadx  = par[35]
    
kappa    = np.array([0.4, 0.4, 0.8, 0.8, 0.8, 0.8])
rho      = np.array([0.0001, 0.001, 0.1, 0.3, 0.4, 0.5])
sigma    = 0.10
phi      = 0.20
eta      = 0.40
theta    = np.array([9.16091458e-06, 9.16091458e-06, 3.95604731e-04, 6.58625427e-05, 4.96556745e-03, 2.36730805e-02])
gamma    = np.array([0.0, 0.0, 0.0, 3.13666771e-04, 0.0, 1.60165839e-04])
nu       = np.array([0.0, 0.0, 0.00, 2.13666771e-04, 1.87118193e-02, 3.36108712e-02])

period  = P1, P3, P4, P5, P9
# params  = c, alphaa, alphaax, betaa, betaax, epsilona, deltaa, deltaax, thetaa, thetaax , alphab, alphabx, betab, betabx, epsilonb, deltab, deltabx, thetab, thetabx, alphac, alphacx, betac, betacx, epsilonc, deltac, deltacx, thetac, thetacx, alphad, alphadx, betad, betadx, epsilond, deltad, deltadx, thetad, thetadx


### Initial States 
N = 83166711
N = N * np.array([0.0470, 0.0920, 0.2280, 0.3500, 0.2150, 0.0680])#np.array([0.0476, 0.0893, 0.2299, 0.3477, 0.2171, 0.0683]) #np.array([0.0477, 0.0903, 0.2275, 0.3447, 0.2183, 0.0714])

Sa_0   = pop[0]
Sb_0   = pop[1]
Sc_0   = pop[2]
Sd_0   = pop[3]
Ba_0   = pop[4]
Bb_0   = pop[5]
Vc_0   = pop[6]
Vd_0   = pop[7]
Eax_0  = pop[8]
Eay_0  = pop[9]
Ebx_0  = pop[10]
Eby_0  = pop[11]
Ecx_0  = pop[12]
Ecy_0  = pop[13]
Edx_0  = pop[14]
Edy_0  = pop[15]
Iax_0  = pop[16]
Iay_0  = pop[17]
Ibx_0  = pop[18]
Iby_0  = pop[19]
Icx_0  = pop[20]
Icy_0  = pop[21]
Idx_0  = pop[22]
Idy_0  = pop[23]
Hax_0  = pop[24]
Hay_0  = pop[25]
Hbx_0  = pop[26]
Hby_0  = pop[27]
Hcx_0  = pop[28]
Hcy_0  = pop[29]
Hdx_0  = pop[30]
Hdy_0  = pop[31]
Uax_0  = pop[32]
Uay_0  = pop[33]
Ubx_0  = pop[34]
Uby_0  = pop[35]
Ucx_0  = pop[36]
Ucy_0  = pop[37]
Udx_0  = pop[38]
Udy_0  = pop[39]
R_0    = pop[40]
D_0    = pop[41]
Ns_0   = idata[-18]/N
Nh_0   = hdata[-16]/N
Nu_0   = udata[-16]/N


dum = 40*7
ft  = np.linspace(0, dum, int(dum/dt)+1)
init_vals = Sa_0, Sb_0, Sc_0, Sd_0, Ba_0, Bb_0, Vc_0, Vd_0, Eax_0, Eay_0, Ebx_0, Eby_0, Ecx_0, Ecy_0, Edx_0, Edy_0, Iax_0, Iay_0, Ibx_0, Iby_0, Icx_0, Icy_0, Idx_0, Idy_0, Hax_0, Hay_0, Hbx_0, Hby_0, Hcx_0, Hcy_0, Hdx_0, Hdy_0, Uax_0, Uay_0, Ubx_0, Uby_0, Ucx_0, Ucy_0, Udx_0, Udy_0, R_0, D_0, Ns_0, Nh_0, Nu_0
fSa, fSb, fSc, fSd, fBa, fBb, fVc, fVd, fEax, fEay, fEbx, fEby, fEcx, fEcy, fEdx, fEdy, fIax, fIay, fIbx, fIby, fIcx, fIcy, fIdx, fIdy, fHax, fHay, fHbx, fHby, fHcx, fHcy, fHdx, fHdy, fUax, fUay, fUbx, fUby, fUcx, fUcy, fUdx, fUdy, fR, fD, fNs, fNh, fNu = init_vals
   
for i in range(40):
    
    dummyt = np.linspace(0, 7, 8)
        
    betaa = betaa * coef[i] 
    betab = betab * coef[i]
    betac = betac * coef[i] 
    betad = betad * coef[i]
    
    alphaa = alphaa * coeh[i]
    alphab = alphab * coeh[i]
    alphac = alphac * coeh[i]
    alphad = alphad * coeh[i]
    
    deltaa = deltaa * coeu[i]
    deltab = deltab * coeu[i]
    deltac = deltac * coeu[i]
    deltad = deltad * coeu[i]
    
    thetaa = thetaa * coed[i]
    thetab = thetab * coed[i]
    thetac = thetac * coed[i]
    thetad = thetad * coed[i]
    

    betaax  = betaa * 0.3
    alphaax = alphaa * 0.2
    deltaax = deltaa * 0.2
    thetaax = thetaa * 0.2
    betabx  = betab * 0.3
    alphabx = alphab * 0.2
    deltabx = deltab * 0.2
    thetabx = thetab * 0.2
    betacx  = betac * 0.3
    alphacx = alphac * 0.3
    deltacx = deltac * 0.3
    thetacx = thetac * 0.3
    betadx  = betad * 0.3
    alphadx = alphad * 0.3
    deltadx = deltad * 0.3
    thetadx = thetad * 0.3
    
    params   = c, alphaa, alphaax, betaa, betaax, epsilona, deltaa, deltaax, thetaa, thetaax , alphab, alphabx, betab, betabx, epsilonb, deltab, deltabx, thetab, thetabx, alphac, alphacx, betac, betacx, epsilonc, deltac, deltacx, thetac, thetacx, alphad, alphadx, betad, betadx, epsilond, deltad, deltadx, thetad, thetadx

    # if i >= 10:
    #     betaai = betaa * 1.1
    #     betabi = betab * 1.1
    #     betaci = betac * 1.0
    #     betadi = betad * 1.0
        
    #     alphaai = alphaa * 1.05
    #     alphabi = alphab * 1.05
    #     alphaci = alphac * 1.05
    #     alphadi = alphad * 1.0
        
    #     deltaai = deltaa * 1.05
    #     deltabi = deltab * 1.05
    #     deltaci = deltac * 1.05
    #     deltadi = deltad * 1.0
        
    #     thetaai = thetaa * 1.05
    #     thetabi = thetab * 1.05
    #     thetaci = thetac * 1.05
    #     thetadi = thetad * 1.0
        
    #     params   = c, alphaai, alphaax, betaai, betaax, epsilona, deltaai, deltaax, thetaai, thetaax , alphabi, alphabx, betabi, betabx, epsilonb, deltabi, deltabx, thetabi, thetabx, alphaci, alphacx, betaci, betacx, epsilonc, deltaci, deltacx, thetaci, thetacx, alphadi, alphadx, betadi, betadx, epsilond, deltadi, deltadx, thetadi, thetadx

    nSa, nSb, nSc, nSd, nBa, nBb, nVc, nVd, nEax, nEay, nEbx, nEby, nEcx, nEcy, nEdx, nEdy, nIax, nIay, nIbx, nIby, nIcx, nIcy, nIdx, nIdy, nHax, nHay, nHbx, nHby, nHcx, nHcy, nHdx, nHdy, nUax, nUay, nUbx, nUby, nUcx, nUcy, nUdx, nUdy, nR, nD, nNs, nNh, nNu = model(0, init_vals, period, params, dummyt)
    init_vals =  nSa[-1], nSb[-1], nSc[-1], nSd[-1], nBa[-1], nBb[-1], nVc[-1], nVd[-1], nEax[-1], nEay[-1], nEbx[-1], nEby[-1], nEcx[-1], nEcy[-1], nEdx[-1], nEdy[-1], nIax[-1], nIay[-1], nIbx[-1], nIby[-1], nIcx[-1], nIcy[-1], nIdx[-1], nIdy[-1], nHax[-1], nHay[-1], nHbx[-1], nHby[-1], nHcx[-1], nHcy[-1], nHdx[-1], nHdy[-1], nUax[-1], nUay[-1], nUbx[-1], nUby[-1], nUcx[-1], nUcy[-1], nUdx[-1], nUdy[-1], nR[-1], nD[-1], nNs[-1], nNh[-1], nNu[-1] 
    
    # if i == 12:
    #     epsilona = np.array([0., 0.0, 0.35, 0.35, 0.35, 0.35])
    #     epsilonb = np.array([0., 0.0, 0.35, 0.35, 0.35, 0.35])
    #     epsilonb = np.array([0., 0.14, 0.14, 0.14, 0.14, 0.14])
    #     epsilond = np.array([0., 0.14, 0.14, 0.14, 0.14, 0.14])
        
    #     params = c, alphaa, alphaax, betaa, betaax, epsilona, deltaa, deltaax, thetaa, thetaax , alphab, alphabx, betab, betabx, epsilonb, deltab, deltabx, thetab, thetabx, alphac, alphacx, betac, betacx, epsilonc, deltac, deltacx, thetac, thetacx, alphad, alphadx, betad, betadx, epsilond, deltad, deltadx, thetad, thetadx

    fSa  = np.vstack((fSa, nSa[1:,:]))
    fSb  = np.vstack((fSb, nSb[1:,:]))
    fSc  = np.vstack((fSc, nSc[1:,:]))
    fSd  = np.vstack((fSd, nSd[1:,:]))
    fBa  = np.vstack((fBa, nBa[1:,:]))
    fBb  = np.vstack((fBb, nBb[1:,:]))
    fVc  = np.vstack((fVc, nVc[1:,:]))
    fVd  = np.vstack((fVd, nVd[1:,:]))
    fEax = np.vstack((fEax, nEax[1:,:]))
    fEay = np.vstack((fEay, nEay[1:,:]))
    fEbx = np.vstack((fEbx, nEbx[1:,:]))
    fEby = np.vstack((fEby, nEby[1:,:]))
    fEcx = np.vstack((fEcx, nEcx[1:,:]))
    fEcy = np.vstack((fEcy, nEcy[1:,:]))
    fEdx = np.vstack((fEdx, nEdx[1:,:]))
    fEdy = np.vstack((fEdy, nEdy[1:,:]))
    fIax = np.vstack((fIax, nIax[1:,:]))
    fIay = np.vstack((fIay, nIay[1:,:]))
    fIbx = np.vstack((fIbx, nIbx[1:,:]))
    fIby = np.vstack((fIby, nIby[1:,:]))
    fIcx = np.vstack((fIcx, nIcx[1:,:]))
    fIcy = np.vstack((fIcy, nIcy[1:,:]))
    fIdx = np.vstack((fIdx, nIdx[1:,:]))
    fIdy = np.vstack((fIdy, nIdy[1:,:]))
    fHax = np.vstack((fHax, nHax[1:,:]))
    fHay = np.vstack((fHay, nHay[1:,:]))
    fHbx = np.vstack((fHbx, nHbx[1:,:]))
    fHby = np.vstack((fHby, nHby[1:,:]))
    fHcx = np.vstack((fHcx, nHcx[1:,:]))
    fHcy = np.vstack((fHcy, nHcy[1:,:]))
    fHdx = np.vstack((fHdx, nHdx[1:,:]))
    fHdy = np.vstack((fHdy, nHdy[1:,:]))
    fUax = np.vstack((fUax, nUax[1:,:]))
    fUay = np.vstack((fUay, nUay[1:,:]))
    fUbx = np.vstack((fUbx, nUbx[1:,:]))
    fUby = np.vstack((fUby, nUby[1:,:]))
    fUcx = np.vstack((fUcx, nUcx[1:,:]))
    fUcy = np.vstack((fUcy, nUcy[1:,:]))
    fUdx = np.vstack((fUdx, nUdx[1:,:]))
    fUdy = np.vstack((fUdy, nUdy[1:,:]))
    fR  = np.vstack((fR, nR[1:,:]))
    fD  = np.vstack((fD, nD[1:,:]))
    fNs  = np.vstack((fNs, nNs[1:,:]))
    fNh  = np.vstack((fNh, nNh[1:,:]))
    fNu  = np.vstack((fNu, nNu[1:,:]))
    
zN = fNs[::7]*N
zH = fNh[::7]*N
zU = fNu[::7]*N
zD = fD[::7]*N
zDD = np.zeros((40,6))
for i in range(40):
    zDD[i] = zD[i+1] - zD[i]

