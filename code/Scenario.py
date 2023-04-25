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

def model(opparams, init_vals, period, params, t, index):
    S_0, V_0, B_0, E_0, Ev_0, Eb_0, I_0, A_0, Iv_0, Ib_0, H_0, Hv_0, Hb_0, U_0, Uv_0, Ub_0, L_0, F_0, R_0, D_0, Ns_0, Nh_0, Nu_0 = init_vals
    S, V, B, E, Ev, Eb, I, A, Iv, Ib, H, Hv, Hb, U, Uv, Ub, L, F, R, D, Ns, Nh, Nu = [S_0], [V_0], [B_0], [E_0], [Ev_0], [Eb_0], [I_0], [A_0], [Iv_0], [Ib_0], [H_0], [Hv_0], [Hb_0], [U_0], [Uv_0], [Ub_0], [L_0], [F_0], [R_0], [D_0], [Ns_0], [Nh_0], [Nu_0]
    P1, P2, P3, P4, P5, P6, P7, P8, P9 = period
    c, kappa, rho, delta, sigma, phi, gamma, eta, nu, epsilon, epsilonv = params
    
    beta  = opparams[:6]    
    theta = opparams[6:12]  
    alpha = opparams[12:18] 
    delta = opparams[18:]
    
    betav  = beta * np.array([0.0, 0.2 , 0.5 , 0.5, 0.6, 0.6])
    betab  = beta * np.array([0.0, 0.0 , 0.4, 0.4, 0.25, 0.25]) # normal case
    
    thetav = theta * np.array([0.0, 0.2 , 0.5 , 0.5, 0.6, 0.6])
    thetab = theta * np.array([0.0, 0.0 , 0.4, 0.4, 0.25, 0.25]) # normal case
    
    alphav = alpha * np.array([0.0, 0.2 , 0.5 , 0.5, 0.6, 0.6])
    alphab = alpha * np.array([0.0, 0.0 , 0.4, 0.4, 0.25, 0.25]) # normal case
    
    deltav = delta * np.array([0.0, 0.2 , 0.5 , 0.5, 0.6, 0.6])
    deltab = delta * np.array([0.0, 0.0 , 0.4, 0.4, 0.25, 0.25]) # normal case
    
    # if index >= 10:
        # ###NO-VACCINE 
        # betav  = betav * 1.1
        # thetav = thetav * 1.1
        # alphav = alphav * 1.1
        # deltav = deltav * 1.1
        
        # betab  = betab * 1.1
        # thetab = thetab * 1.1
        # alphab = alphab * 1.1
        # deltab = deltab * 1.1
        
        # ### VACCINE 
        # betav  = betav * 1.7
        # thetav = thetav * .5
        # alphav = alphav * .5
        # deltav = deltav * .5
        
        # betab  = betab * 1.7
        # thetab = thetab * .5
        # alphab = alphab * .5
        # deltab = deltab * .5
    
    dt = (t[1] - t[0])*1./7
    
    for i in t[1:]:
        
        next_Ns = c.dot((I[-1]+A[-1]+Iv[-1]+Ib[-1])*N)*beta*S[-1] +  c.dot((I[-1]+A[-1]+Iv[-1]+Ib[-1])*N)*betav*V[-1] + c.dot((I[-1]+A[-1]+Iv[-1]+Ib[-1])*N)*betab*B[-1]
        next_Nu = (delta*P4*H[-1] + deltav*P4*Hv[-1] + deltab*P4*Hb[-1])
        
        
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


def cost(opparams, modelparams, idata, cddata, hdata, i):
    init_vals, period, params, N, t = modelparams
    S, V, B, E, Ev, Eb, I, A, Iv, Ib, H, Hv, Hb, U, Uv, Ub, L, F, R, D, Ns, Nh = model(opparams, init_vals, period, params, t)
    dum = 0
    iw = 1
    for j in range(iw):
        simi = Ns[(iw - j)*7,:] * N
        q = (simi - idata[i-j])**2 * 1./max(idata[i-j])
        simc = D[(iw - j)*7,:] * N
        p = (simc - cddata[i-j])**2 * 1./max(cddata[i-j])
        simh = Nh[(iw - j)*7,:] * N
        r = (simh - hdata[i-j])**2 * 1./max(hdata[i-j])
        dum = dum + sum(p) + sum(q) + sum(r)
    return dum
   


file = pd.ExcelFile('Forecast.xlsx')
df1 = file.parse(0) #Infection
df2 = file.parse(1) #CumDeath
df3 = file.parse(2) #Death
df4 = file.parse(3) #Hospitalization
df9 = file.parse(4) #ICU

idata  = df1.values[0:, 1:7]
cddata = df2.values[0:, 1:7]
ddata  = df2.values[0:, 8]
dd     = df3.values[0:, 1:7]
hdata  = df4.values[0:, 1:7]
udata  = df9.values[0:, 1:7]
   

idt = idata * (1 - np.array([0., 0.03, 0.15, 0.32, 0.55, 0.55]))
vdt = idata * np.array([0., 0.03, 0.15, 0.32, 0.55, 0.55])

it = 122
t_max = 7*it
times = np.arange(0, t_max+1, 7)

dt = 1
t  = np.linspace(0, t_max, int(t_max/dt)+1)

### Parameters
P1 = 1./3
P2 = 1./7
P3 = 1./4
P4 = 1./7
P5 = 1./10
P6 = 1./14
P7 = 1./90
P8 = 1./360
P9 = 1./14

c   = np.array([[5.090928e-07, 1.016800e-07, 9.904523e-08, 6.037230e-08, 3.330160e-08, 1.418771e-08],
                [1.016800e-07, 8.277460e-07, 7.586532e-08, 7.709942e-08, 3.231887e-08, 1.384713e-08],
                [9.904523e-08, 7.586532e-08, 2.650871e-07, 1.042291e-07, 3.812121e-08, 2.656602e-08],
                [6.037230e-08, 7.709942e-08, 1.042291e-07, 1.308419e-07, 5.894746e-08, 3.369221e-08],
                [3.330160e-08, 3.231887e-08, 3.812121e-08, 5.894746e-08, 1.187878e-07, 6.106433e-08],
                [1.418771e-08, 1.384713e-08, 2.656602e-08, 3.369221e-08, 6.106433e-08, 8.840770e-08]])

kappa    = np.array([0.4, 0.4, 0.8, 0.8, 0.8, 0.8])
rho      = np.array([0.0001, 0.001, 0.1, 0.3, 0.4, 0.5])
alpha    = np.array([0.025, 0.003, 0.012, 0.025, 0.099, 0.262])
delta    = np.array([0.01, 0.05, 0.063, 0.376, 0.545, 0.709]) #np.array([0.005 , 0.015 , 0.025, 0.188 , 0.2725, 0.3545])#0.5 * np.array([0.01, 0.05, 0.063, 0.376, 0.545, 0.709])#np.array([0.05, 0.05, 0.063, 0.122, 0.303, 0.709])
sigma    = 0.10
phi      = 0.20
eta      = 0. #0.40
beta     = np.array([0.0474986, 0.09716402, 0.07156201, 0.0664224, 0.04255577, 0.0645]) #0.6*np.array([0.02691644, 0.06744003, 0.09427002, 0.1285704 , 0.15575962, 0.85])#5 * np.array([0.01493689, 0.0187243 , 0.0197207 , 0.01965221, 0.01581639, 0.02185509])
theta    = np.array([9.16091458e-06, 9.16091458e-06, 3.95604731e-04, 6.58625427e-05, 4.96556745e-03, 2.36730805e-02])
gamma    = np.array([0.0, 0.0, 0.0, 3.13666771e-04, 0.0, 1.60165839e-04])
nu       = np.array([0.0, 0.0, 0.00, 2.13666771e-05, 1.87118193e-04, 8.36108712e-03])
epsilon  = np.array([0., 0., 0., 0., 0., 0.]) #np.array([0., 0.01, 0.01, 0.01, 0.01, 0.01])
betav    = beta * np.array([0.0, 0.01 , 0.5 , 0.5, 0.6, 0.6])
thetav   = theta * np.array([0.0, 0.0 , 0.5 , 0.5, 0.4, 0.4]) #np.array([0, 0, 3.9e-05, 6.5e-06, 4.9e-03, 2.3e-02])
deltav   = delta * np.array([0.0, 0.0 , 0.5 , 0.5, 0.4, 0.4]) #np.array([0, 0, 0.01, 0.01, 0.1, 0.1])
alphav   = alpha * np.array([0., 0.001, 0.4, 0.4, 0.3, 0.2])
epsilonv = np.array([0., 0., 0., 0., 0., 0.]) #np.array([0., 0., 0., 0.001, 0.005, 0.005])
betab    = beta * np.array([0.0, 0.0 , 0.4, 0.4, 0.25, 0.25])
thetab   = theta * np.array([0.0, 0.0 , 0.4 , 0.4, 0.1, 0.1])
deltab   = delta * np.array([0.0, 0.0 , 0.4 , 0.4, 0.1, 0.1])
alphab   = alpha * np.array([0.0, 0.0 , 0.25 , 0.2, 0.2, 0.1])

period  = P1, P2, P3, P4, P5, P6, P7, P8, P9
params  = c, kappa, rho, delta, sigma, phi, gamma, eta, nu, epsilon, deltav, epsilonv, deltab

### Initial States KW 14
N = 83166711         
e = 185790           
i = 85778
a = i
h = 0.14*i           
u = 228
l = 0.3*i            
r = 100		         
f = 26400
d = 1158

N = N * np.array([0.0470, 0.0920, 0.2280, 0.3500, 0.2150, 0.0680])
i = np.array([4.81e-05, 7.70e-05, 4.04e-04, 5.15e-04, 2.57e-04, 1.63e-04])

E_0  = np.array([0.00195301, 0.00458122, 0.00655739, 0.00594781, 0.00378435, 0.00257799])
I_0  = np.array([0.00379192, 0.00869907, 0.00663173, 0.00574209, 0.00285751, 0.00205869])
A_0  = np.array([4.56E-03,	1.04E-02,	3.70E-03,	3.11E-03,	8.18E-04,	6.23E-04])
H_0  = np.array([5.82782276e-05, 1.20867758e-05, 2.94061651e-05, 3.38267212e-05, 1.14735084e-04, 4.12544870e-04])
U_0  = np.array([1.21775401e-05, 1.44952436e-05, 1.00052684e-05, 3.22148125e-05, 1.57518302e-04, 2.86893635e-04])
L_0  = np.array([3.73E-06,	6.73E-05,	5.81E-03,	9.33E-03,	4.92E-03,	6.91E-03])
F_0  = np.array([1.82E-01,	4.14E-01,	3.56E-01,	2.69E-01,	1.24E-01,	1.04E-01])
R_0  = np.array([3.53E-06,	6.56E-05,	5.17E-03,	7.98E-03,	3.62E-03,	4.15E-03])
D_0  = np.array([8.96397145e-06, 4.08555359e-06, 2.14639705e-05, 2.54161335e-04, 2.44363422e-03, 1.63754930e-02])
V_0  = np.array([0.00E+00,	1.65E-01,	2.09E-01,	1.40E-01,	4.37E-01,	5.26E-01])
Ev_0 = np.array([0.00091906, 0.00215587, 0.00308583, 0.00279897, 0.00178087, 0.00121317])
Iv_0 = np.array([0.00178443, 0.00409368, 0.00312081, 0.00270216, 0.00134471, 0.00096879])
Hv_0 = np.array([2.74250483e-05, 5.68789450e-06, 1.38381953e-05, 1.59184570e-05, 5.39929805e-05, 1.94138762e-04])
Uv_0 = np.array([5.73060710e-06, 6.82129113e-06, 4.70836159e-06, 1.51599118e-05, 7.41262596e-05, 1.35008769e-04])
B_0  = np.zeros((6))
Eb_0 = np.zeros((6))
Ib_0 = np.zeros((6))
Hb_0 = np.zeros((6))
Ub_0 = np.zeros((6))
S_0  = N*1./N - (E_0+I_0+A_0+H_0+U_0+L_0+F_0+R_0+D_0+V_0+Ev_0+Iv_0+Hv_0+Uv_0+B_0+Eb_0+Ib_0+Hb_0+Ub_0)
Ns_0   = idata[-18]/N
Nh_0   = hdata[-16]/N
Nu_0   = udata[-16]/N

file1 = pd.ExcelFile('Dum.xlsx')
df4 = file1.parse(3) #Rate
opparams = df4.values[:,:18]
opparams = opparams.astype(float)

init_vals   = S_0, V_0, B_0, E_0, Ev_0, Eb_0, I_0, A_0, Iv_0, Ib_0, H_0, Hv_0, Hb_0, U_0, Uv_0, Ub_0, L_0, F_0, R_0, D_0, Ns_0, Nh_0, Nu_0
modelparams = init_vals, period, params, N, t
S, V, B, E, Ev, Eb, I, A, Iv, Ib, H, Hv, Hb, U, Uv, Ub, L, F, R, D, Ns, Nh, Nu = [S_0], [V_0], [B_0], [E_0], [Ev_0], [Eb_0], [I_0], [A_0], [Iv_0], [Ib_0], [H_0], [Hv_0], [Hb_0], [U_0], [Uv_0], [Ub_0], [L_0], [F_0], [R_0], [D_0], [Ns_0], [Nh_0], [Nu_0]
    

file1 = pd.ExcelFile('Dum.xlsx')
df5 = file1.parse(2) #Coef
df4 = file1.parse(3) #Rate
    
sce = 1
coef = df5.values[1:,1+4*(sce-1)]
coef = coef.astype(float)
coeh = df5.values[1:,2+4*(sce-1)]
coeh = coeh.astype(float)
coeu = df5.values[1:,3+4*(sce-1)]
coeu = coeu.astype(float)
coed = df5.values[1:,4+4*(sce-1)]
coed = coed.astype(float)
    
par = np.zeros(24)
par[:6]    = np.array([0.0372407, 0.0773586, 0.15361, 0.1649, 0.165262, 0.211702])
par[6:12]  = np.array([0.00099942, 0.00096026, 0.085286, 0.188,	0.504,	2.07])
par[12:18] = np.array([0.01372, 0.001194, 0.0029258, 0.004, 0.025, 0.136])
par[18:]   = np.array([0.0734089, 0.721375, 0.437597, 1.42857, 1.76, 0.883601])	

epsilon  = np.array([0., 0.01, 0.05, 0.01, 0.05, 0.05])
epsilonv = np.array([0., 0., 0.04, 0.04, 0.05, 0.05])


params   = c, kappa, rho, delta, sigma, phi, gamma, eta, nu, epsilon, epsilonv

dum = 40*7
ft  = np.linspace(0, dum, int(dum/dt)+1)  

# init_vals = S[-1], V[-1], B[-1], E[-1], Ev[-1], Eb[-1], I[-1], A[-1], Iv[-1], Ib[-1], H[-1], Hv[-1], Hb[-1], U[-1], Uv[-1], Ub[-1], L[-1], F[-1], R[-1], D[-1], idata[-3]/N, hdata[-3]/N
fS, fV, fB, fE, fEv, fEb, fI, fA, fIv, fIb, fH, fHv, fHb, fU, fUv, fUb, fL, fF, fR, fD, fNs, fNh, fNu = init_vals
for i in range(40):
    dummyt = np.linspace(0, 7, 8)
    par[:6]   = par[:6]*coef[i]
    par[6:12] = par[6:12]*coed[i]
    par[12:18] = par[12:18]*coeh[i]
    par[18:] = par[18:]*coeu[i]
    # par[par>1] = 1
    
    if i == 12:
        epsilon  = np.array([0.0, 0.90 , 0.90, 0.90, 0.90, 0.90])
        epsilonv  = np.array([0.0, 0.0 , 0.35, 0.35, 0.35, 0.35])
        ## epsilon = np.array([0.0, 0.14 , 0.14, 0.14, 0.14, 0.14])
        ## epsilonv = np.array([0.0, 0.0 , 0.14, 0.14, 0.14, 0.14])
        params  = c, kappa, rho, delta, sigma, phi, gamma, eta, nu, epsilon, epsilonv

    nS, nV, nB, nE, nEv, nEb, nI, nA, nIv, nIb, nH, nHv, nHb, nU, nUv, nUb, nL, nF, nR, nD, nNs, nNh, nNu = model(par, init_vals, period, params, dummyt, i)
    init_vals = nS[-1], nV[-1], nB[-1], nE[-1], nEv[-1], nEb[-1], nI[-1], nA[-1], nIv[-1], nIb[-1], nH[-1], nHv[-1], nHb[-1], nU[-1], nUv[-1], nUb[-1], nL[-1], nF[-1], nR[-1], nD[-1], nNs[-1], nNh[-1], nNu[-1]

    fS  = np.vstack((fS, nS[1:,:]))
    fV  = np.vstack((fV, nV[1:,:]))
    fB  = np.vstack((fB, nB[1:,:]))
    fE  = np.vstack((fE, nE[1:,:]))
    fEv = np.vstack((fEv, nEv[1:,:]))
    fEb = np.vstack((fEb, nEb[1:,:]))
    fI  = np.vstack((fI, nI[1:,:]))
    fA  = np.vstack((fA, nA[1:,:]))
    fIv = np.vstack((fIv, nIv[1:,:]))
    fIb = np.vstack((fIb, nIb[1:,:]))
    fH  = np.vstack((fH, nH[1:,:]))
    fHv = np.vstack((fHv, nHv[1:,:]))
    fHb = np.vstack((fHb, nHb[1:,:]))
    fU  = np.vstack((fU, nU[1:,:]))
    fUv = np.vstack((fUv, nUv[1:,:]))
    fUb = np.vstack((fUb, nUb[1:,:]))
    fL  = np.vstack((fL, nL[1:,:]))
    fF  = np.vstack((fF, nF[1:,:]))
    fR  = np.vstack((fR, nR[1:,:]))
    fD  = np.vstack((fD, nD[1:,:]))
    fNs = np.vstack((fNs, nNs[1:,:]))
    fNh = np.vstack((fNh, nNh[1:,:]))
    fNu = np.vstack((fNu, nNu[1:,:]))

zN = fNs[::7]*N
zE = fE[::7]*P3*N
zU = fNu[::7]*N
zH = fNh[::7]*N
zD = fD[::7]*N
zDD = np.zeros((40,6))
coefu = np.array([35, 83, 291, 1439, 4266, 2492])/zU[1]
for i in range(40):
    zDD[i] = zD[i+1] - zD[i]
    