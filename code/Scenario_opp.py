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
bounds = Bounds(np.zeros((18)), 1*np.ones((18)))

def model(opparams, init_vals, period, params, t):
    S_0, V_0, B_0, E_0, Ev_0, Eb_0, I_0, A_0, Iv_0, Ib_0, H_0, Hv_0, Hb_0, U_0, Uv_0, Ub_0, L_0, F_0, R_0, D_0, Ns_0, Nh_0, Nu_0 = init_vals
    S, V, B, E, Ev, Eb, I, A, Iv, Ib, H, Hv, Hb, U, Uv, Ub, L, F, R, D, Ns, Nh, Nu = [S_0], [V_0], [B_0], [E_0], [Ev_0], [Eb_0], [I_0], [A_0], [Iv_0], [Ib_0], [H_0], [Hv_0], [Hb_0], [U_0], [Uv_0], [Ub_0], [L_0], [F_0], [R_0], [D_0], [Ns_0], [Nh_0], [Nu_0]
    P1, P2, P3, P4, P5, P6, P7, P8, P9 = period
    c, kappa, rho, sigma, phi, gamma, eta, nu, delta, epsilon, epsilonv = params
    
    beta  = opparams[:6] 
    theta = opparams[6:12]
    alpha = opparams[12:18]
    
    betav  = beta * np.array([1.0, 0.2 , 0.5 , 0.5, 0.6, 0.6])
    betab  = beta * np.array([1.0, 1.0 , 0.4, 0.4, 0.25, 0.25]) 
    
    thetav = theta * np.array([1.0, 0.1 , 0.3 , 0.3, 0.4, 0.4])
    thetab = theta * np.array([1.0, 1.0 , 0.15, 0.15, 0.25, 0.25]) 
    
    alphav = alpha * np.array([1.0, 0.1 , 0.3 , 0.3, 0.4, 0.4])
    alphab = alpha * np.array([1.0, 1.0 , 0.15, 0.15, 0.25, 0.25]) 
    
    deltav = delta * np.array([1.0, 0.2 , 0.5 , 0.5, 0.6, 0.6])
    deltab = delta * np.array([1.0, 1.0 , 0.4, 0.4, 0.25, 0.25]) 
    
    dt = (t[1] - t[0])*1./7
    
    for i in t[1:]:
        
        next_Ns = Ns[-1] + (c.dot((I[-1]+A[-1]+Iv[-1]+Ib[-1])*N)*beta*S[-1] +  c.dot((I[-1]+A[-1]+Iv[-1]+Ib[-1])*N)*betav*V[-1] + c.dot((I[-1]+A[-1]+Iv[-1]+Ib[-1])*N)*betab*B[-1])*dt
        next_Nu = Nu[-1] + (delta*P4*H[-1] + deltav*P4*Hv[-1] + deltab*P4*Hb[-1])*dt
        
        next_S  = S[-1] - (c.dot(I[-1]*N)*beta*S[-1] + c.dot(A[-1]*N)*beta*S[-1] + c.dot(Iv[-1]*N)*beta*S[-1] + c.dot(Ib[-1]*N)*beta*S[-1] + epsilon*P9*S[-1] - P7*R[-1] - P8*F[-1])*dt
        next_V  = V[-1] - (c.dot(I[-1]*N)*betav*V[-1] + c.dot(A[-1]*N)*betav*V[-1] + c.dot(Iv[-1]*N)*betav*V[-1] + c.dot(Ib[-1]*N)*betav*V[-1] - epsilon*P9*S[-1] + epsilonv*P9*V[-1])*dt
        next_B  = B[-1] - (c.dot(I[-1]*N)*betab*B[-1] + c.dot(A[-1]*N)*betab*B[-1] + c.dot(Iv[-1]*N)*betab*B[-1] + c.dot(Ib[-1]*N)*betab*B[-1] - epsilonv*P9*V[-1])*dt
        next_E  = E[-1] + (c.dot(I[-1]*N)*beta*S[-1] + c.dot(A[-1]*N)*beta*S[-1] + c.dot(Iv[-1]*N)*beta*S[-1] + c.dot(Ib[-1]*N)*beta*S[-1] - P1*E[-1])*dt
        next_Ev = Ev[-1] + (c.dot(I[-1]*N)*betav*V[-1] + c.dot(A[-1]*N)*betav*V[-1] + c.dot(Iv[-1]*N)*betav*V[-1] + c.dot(Ib[-1]*N)*betav*V[-1] - P1*Ev[-1])*dt
        next_Eb = Eb[-1] + (c.dot(I[-1]*N)*betav*V[-1] + c.dot(A[-1]*N)*betav*V[-1] + c.dot(Iv[-1]*N)*betav*V[-1] + c.dot(Ib[-1]*N)*betab*B[-1] - P1*Eb[-1])*dt
        next_I  = I[-1] + (kappa*P1*E[-1] - alpha*P3*I[-1] - (1-alpha)*(1-rho)*P3*I[-1] - (1-alpha)*rho*P3*I[-1])*dt
        next_A  = A[-1] + ((1-kappa)*P1*E[-1] - P2*A[-1])*dt
        next_Iv = Iv[-1] + (P1*Ev[-1] - P3*Iv[-1])*dt
        next_Ib = Ib[-1] + (P1*Eb[-1] - P3*Ib[-1])*dt
        next_H  = H[-1] + (alpha*P3*I[-1] - P4*H[-1] + (1-gamma)*eta*P6*L[-1])*dt
        next_Hv = Hv[-1] + (alphav*P3*Iv[-1] - P4*Hv[-1])*dt
        next_Hb = Hb[-1] + (alphab*P3*Ib[-1] - P4*Hb[-1])*dt
        next_U  = U[-1] + (delta*P4*H[-1] - P5*U[-1])*dt
        next_Uv = Uv[-1] + (deltav*P4*Hv[-1] - P5*Uv[-1])*dt
        next_Ub = Ub[-1] + (deltab*P4*Hb[-1] - P5*Ub[-1])*dt
        next_L  = L[-1] + ((1-alpha)*rho*P3*I[-1] + (1-delta)*phi*P4*H[-1] + (1-theta)*sigma*P5*U[-1] - P6*L[-1])*dt
        next_F  = F[-1] + ((1-nu)*P2*A[-1] + (1-alpha)*(1-rho)*P3*I[-1] + (1-delta)*(1-phi)*P4*H[-1] + (1-theta)*(1-sigma)*P5*U[-1] - P8*F[-1] + (1-alphav)*P3*Iv[-1] + (1-deltav)*P4*Hv[-1] +(1-thetav)*P5*Uv[-1] + (1-alphab)*P3*Ib[-1] + (1-deltab)*P4*Hb[-1] +(1-thetab)*P5*Ub[-1])*dt
        next_R  = R[-1] + ((1-gamma)*(1-eta)*P6*L[-1] - P7*R[-1])*dt
        next_D  = D[-1] + (gamma*P6*L[-1] + theta*P5*U[-1] + nu*P2*A[-1] + thetav*P5*Uv[-1] + thetab*P5*Ub[-1])*dt
        
        next_Nh = next_H + next_Hv + next_Hb
        
        Ns = np.vstack((Ns, next_Ns))
        Nh = np.vstack((Nh, next_Nh))
        Nu = np.vstack((Nu, next_Nu))
        
        S  = np.vstack((S, next_S))
        V  = np.vstack((V, next_V))
        B  = np.vstack((B, next_B))
        E  = np.vstack((E, next_E))
        Ev = np.vstack((Ev, next_Ev))
        Eb = np.vstack((Eb, next_Eb))
        I  = np.vstack((I, next_I))
        A  = np.vstack((A, next_A))
        Iv = np.vstack((Iv, next_Iv))
        Ib = np.vstack((Ib, next_Ib))
        H  = np.vstack((H, next_H))
        Hv = np.vstack((Hv, next_Hv))
        Hb = np.vstack((Hb, next_Hb))
        U  = np.vstack((U, next_U))
        Uv = np.vstack((Uv, next_Uv))
        Ub = np.vstack((Ub, next_Ub))
        L  = np.vstack((L, next_L))
        F  = np.vstack((F, next_F))
        R  = np.vstack((R, next_R))
        D  = np.vstack((D, next_D))
    
    return S, V, B, E, Ev, Eb, I, A, Iv, Ib, H, Hv, Hb, U, Uv, Ub, L, F, R, D, Ns, Nh, Nu


def cost(opparams, modelparams, idata, cddata, hdata, udata, i):
    init_vals, period, params, N, t = modelparams
    S, V, B, E, Ev, Eb, I, A, Iv, Ib, H, Hv, Hb, U, Uv, Ub, L, F, R, D, Ns, Nh, Nu = model(opparams, init_vals, period, params, t)
    
    dum = 0
    
    dui  = sum(idata[:i+1])
    simi = Ns[7,:] * N
    p    = (simi - dui)**2 * 1./max(dui)
    
    simh = Nh[7,:] * N
    q    = (simh - hdata[i])**2 * 1./max(hdata[i])
    
    # duu  = sum(udata[:i+1])
    # simu = Nu[7,:] * N
    # r    = (simu - duu)**2 * 1./max(duu)
    
    simc = D[7,:] * N
    x = (simc - cddata[i])**2 * 1./max(cddata[i])
    
    dum = dum + sum(p) + sum(q) + sum(x) 
    return dum
   

file = pd.ExcelFile('Forecast.xlsx')
df1  = file.parse(0) #Infection
df2  = file.parse(1) #CumDeath
df3  = file.parse(2) #Death
df4  = file.parse(3) #Hospitalization
df5  = file.parse(4) #ICU

idata  = df1.values[40:149, 1:7]
cddata = df2.values[40:149, 1:7]
ddata  = df3.values[40:149, 1:7]
hdata  = df4.values[40:149, 1:7]
udata  = df5.values[40:149, 1:7]
   

it = 109
t_max = 7*it
times = np.arange(0, t_max+1, 7)

dt = 1
t  = np.linspace(0, t_max, int(t_max/dt)+1)

### Parameters
P1 = 1./3
P2 = 1./7
P3 = 1./4
P4 = 1./7
P5 = 1./5
P6 = 1./14
P7 = 1./90
P8 = 1./200
P9 = 1./14

c   = np.array([[2.021E-07, 7.763E-08, 4.520E-08, 4.789E-08, 2.909E-08, 4.994E-08],
                [7.763E-08, 5.721E-08, 2.569E-08, 2.176E-08, 1.171E-08, 2.647E-08],
                [4.520E-08, 2.569E-08, 2.942E-08, 1.908E-08, 1.274E-08, 4.122E-08],
                [4.789E-08, 2.176E-08, 1.908E-08, 1.418E-08, 8.350E-09, 1.447E-08],
                [2.909E-08, 1.171E-08, 1.274E-08, 8.350E-09, 1.488E-08, 1.441E-08],
                [4.994E-08, 2.647E-08, 4.122E-08, 1.447E-08, 1.441E-08, 2.858E-08]])

kappa    = np.array([0.4, 0.4, 0.8, 0.8, 0.8, 0.8])
rho      = np.array([0.0001, 0.001, 0.1, 0.3, 0.4, 0.5])
alpha    = np.array([0.025, 0.003, 0.012, 0.025, 0.099, 0.262])
delta    = np.array([0.01, 0.05, 0.263, 0.476, 0.545, 0.709]) 
sigma    = 0.10
phi      = 0.20
eta      = 0.
beta     = np.array([0.0474986, 0.09716402, 0.07156201, 0.0664224, 0.04255577, 0.0645]) 
theta    = np.array([9.16091458e-06, 9.16091458e-06, 3.95604731e-04, 6.58625427e-05, 4.96556745e-03, 2.36730805e-02])
gamma    = np.array([0.0, 0.0, 0.0, 3.13666771e-04, 0.0, 1.60165839e-04])
nu       = np.array([0.0, 0.0, 0.00, 2.13666771e-05, 1.87118193e-04, 8.36108712e-03])
epsilon  = np.array([0., 0., 0., 0., 0., 0.]) #np.array([0., 0.01, 0.01, 0.01, 0.01, 0.01])
betav    = beta * np.array([0.0, 0.01 , 0.5 , 0.5, 0.6, 0.6])
thetav   = theta * np.array([0.0, 0.0 , 0.5 , 0.5, 0.4, 0.4]) 
deltav   = delta * np.array([0.0, 0.0 , 0.5 , 0.5, 0.4, 0.4]) 
alphav   = alpha * np.array([0., 0.001, 0.4, 0.4, 0.3, 0.2])
epsilonv = np.array([0., 0., 0., 0., 0., 0.]) #np.array([0., 0., 0., 0.001, 0.005, 0.005])
betab    = beta * np.array([0.0, 0.0 , 0.4, 0.4, 0.25, 0.25])
thetab   = theta * np.array([0.0, 0.0 , 0.4 , 0.4, 0.1, 0.1])
deltab   = delta * np.array([0.0, 0.0 , 0.4 , 0.4, 0.1, 0.1])
alphab   = alpha * np.array([0.0, 0.0 , 0.25 , 0.2, 0.2, 0.1])

epsilon  = np.array([0., 0.01, 0.05, 0.01, 0.05, 0.05])
epsilonv = np.array([0., 0., 0.04, 0.04, 0.05, 0.05])

period  = P1, P2, P3, P4, P5, P6, P7, P8, P9
params  = c, kappa, rho, sigma, phi, gamma, eta, nu, delta, epsilon, epsilonv

### Initial States 
N = 83166711         

N = N * np.array([0.0470, 0.0920, 0.2280, 0.3500, 0.2150, 0.0680])

E_0  = 0.7*idata[0]/N 
I_0  = 0.8*idata[0]/N 
A_0  = 0.2*idata[0]/N 
H_0  = hdata[0]/N 
U_0  = np.array([1.21775401e-05, 1.44952436e-05, 1.00052684e-05, 3.22148125e-05, 1.57518302e-04, 2.86893635e-04])
L_0  = np.zeros(6)
F_0  = np.zeros(6)
R_0  = np.zeros(6)
D_0  = cddata[0]/N
V_0  = np.array([0.00E+00,	1.65E-01,	2.09E-01,	1.40E-01,	4.37E-01,	5.26E-01])
Ev_0 = 0.3*idata[0]/N 
Iv_0 = 0.5*I_0 
Hv_0 = np.zeros(6)
Uv_0 = np.zeros(6)
B_0  = np.zeros(6)
Eb_0 = np.zeros(6)
Ib_0 = np.zeros(6)
Hb_0 = np.zeros(6)
Ub_0 = np.zeros(6)
S_0  = N*1./N - (E_0+I_0+A_0+H_0+U_0+L_0+F_0+R_0+D_0+V_0+Ev_0+Iv_0+Hv_0+Uv_0+B_0+Eb_0+Ib_0+Hb_0+Ub_0)
Ns_0 = idata[0]/N
Nh_0 = hdata[0]/N
Nu_0 = udata[0]/N


init_vals   = S_0, V_0, B_0, E_0, Ev_0, Eb_0, I_0, A_0, Iv_0, Ib_0, H_0, Hv_0, Hb_0, U_0, Uv_0, Ub_0, L_0, F_0, R_0, D_0, Ns_0, Nh_0, Nu_0
modelparams = init_vals, period, params, N, t
S, V, B, E, Ev, Eb, I, A, Iv, Ib, H, Hv, Hb, U, Uv, Ub, L, F, R, D, Ns, Nh, Nu = [S_0], [V_0], [B_0], [E_0], [Ev_0], [Eb_0], [I_0], [A_0], [Iv_0], [Ib_0], [H_0], [Hv_0], [Hb_0], [U_0], [Uv_0], [Ub_0], [L_0], [F_0], [R_0], [D_0], [Ns_0], [Nh_0], [Nu_0]

opparams = np.ones(18)*0.001
opp = np.ones((109, 18))*0.001

for i in range(109):
    if i > 0:
        
        dummyt = np.linspace(0, 7, 8)
        modelparams = init_vals, period, params, N, dummyt
        optimizer = opt.minimize(cost, opp[i], args=(modelparams, idata, cddata, hdata, udata, i), tol=1e-10, bounds=bounds)
        opp[i] = optimizer.x
    
        # if i > 3:
        #     epsilonv = np.array([0., 0., 0., 0., 0.025, 0.025])
        #     params   = c, kappa, rho, sigma, phi, gamma, eta, nu, epsilon, epsilonv
            
        # if i > 10:
        #     epsilonv = np.array([0., 0., 0.04, 0.04, 0.05, 0.05])
        #     params   = c, kappa, rho, sigma, phi, gamma, eta, nu, epsilon, epsilonv
            
        # if i > 17:
        #     epsilon  = np.array([0., 0.01, 0.05, 0.01, 0.05, 0.05])
        #     params   = c, kappa, rho, sigma, phi, gamma, eta, nu, epsilon, epsilonv
       
        nS, nV, nB, nE, nEv, nEb, nI, nA, nIv, nIb, nH, nHv, nHb, nU, nUv, nUb, nL, nF, nR, nD, nNs, nNh, nNu = model(opp[i], init_vals, period, params, dummyt)
        init_vals = nS[-1], nV[-1], nB[-1], nE[-1], nEv[-1], nEb[-1], nI[-1], nA[-1], nIv[-1], nIb[-1], nH[-1], nHv[-1], nHb[-1], nU[-1], nUv[-1], nUb[-1], nL[-1], nF[-1], nR[-1], nD[-1], nNs[-1], nNh[-1], nNu[-1]
   
        S  = np.vstack((S, nS[1:,:]))
        V  = np.vstack((V, nV[1:,:]))
        B  = np.vstack((B, nB[1:,:]))
        E  = np.vstack((E, nE[1:,:]))
        Ev = np.vstack((Ev, nEv[1:,:]))
        Eb = np.vstack((Eb, nEb[1:,:]))
        I  = np.vstack((I, nI[1:,:]))
        A  = np.vstack((A, nA[1:,:]))
        Iv = np.vstack((Iv, nIv[1:,:]))
        Ib = np.vstack((Ib, nIb[1:,:]))
        H  = np.vstack((H, nH[1:,:]))
        Hv = np.vstack((Hv, nHv[1:,:]))
        Hb = np.vstack((Hb, nHb[1:,:]))
        U  = np.vstack((U, nU[1:,:]))
        Uv = np.vstack((Uv, nUv[1:,:]))
        Ub = np.vstack((Ub, nUb[1:,:]))
        L  = np.vstack((L, nL[1:,:]))
        F  = np.vstack((F, nF[1:,:]))
        R  = np.vstack((R, nR[1:,:]))
        D  = np.vstack((D, nD[1:,:]))
        Ns = np.vstack((Ns, nNs[1:,:]))
        Nh = np.vstack((Nh, nNh[1:,:]))
        Nu = np.vstack((Nu, nNu[1:,:]))
        

zN = Ns[::7]*N
zH = Nh[::7]*N
zD = D[::7]*N
zNs = np.zeros((109,6))
zDD = np.zeros((109,6))

for i in range(108):
    zDD[i+1] = zD[i+1] - zD[i]
    zNs[i+1] = zN[i+1]-zN[i]
    
zNs[0] = idata[0]
zDD[0] = ddata[0]