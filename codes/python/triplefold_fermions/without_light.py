#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:28:10 2024

@author: aneekphys
"""
import numpy as np
import matplotlib.pyplot as plt
import cmath

N = 200
m = 1
th = 1

iden = np.eye(N)

upper = np.zeros((N,N))

lower = np.zeros((N,N))

for i in range(N-1):
    upper[i][i+1] = 1
    
for i in range(N-1):
    lower[i+1][i] = 1
    
# Without light Lattice Hamiltonian

def Hd(ky,kz,phi):
    Hd_p = np.zeros((3,3),dtype=np.complex128)
    Hd_p[0,2] = np.exp(-1j *phi)* th * np.sin(ky)
    Hd_p[1,2] = np.exp(1j*phi)* m *(np.cos(kz)+ np.cos(ky) -2)
    Hd_p[2,0] = np.exp(1j* phi) *th*np.sin(ky)
    Hd_p[2,1] = np.exp(-1j*phi)*(np.cos(kz) + np.cos(ky) -2)
    return Hd_p

def H_off_d(ky,kz,phi):
    Hl_p = np.zeros((3,3),dtype=np.complex128)
    Hl_p[0,1] = np.exp(1j*phi)*th/(2*1j)
    Hl_p[1,0] = np.exp(-1j*phi)*th/(2*1j)
    Hl_p[1,2] = np.exp(1j*phi) * m/2
    Hl_p[2,1] = np.exp(-1j*phi)*m/2
    Hu_p = np.conjugate(Hl_p.T)
    
    Dict = {}
    
    Dict['Hl'] = Hl_p
    Dict['Hu'] = Hu_p
    
    return Dict
    
def H_full(ky,kz,phi):
    H = np.kron(iden,Hd(ky, kz, phi)) + np.kron(upper,H_off_d(ky, kz, phi)['Hu']) + np.kron(lower,H_off_d(ky, kz, phi)['Hl'])
    return H
    
def eigens(ky,kz,phi):
    E,V = np.linalg.eigh(H_full(ky, kz, phi))
    
    Dict ={}
    Dict['Energy'] = E
    Dict['Eigenvec'] = V
    
    return Dict


ky_list = np.linspace(-np.pi,np.pi,N)
kz_list = np.linspace(-np.pi,np.pi,N)

#####----------- Rimika's Paper Fig-5a reproduce-----------------

phi = 0
kz = np.pi/2-0.001

energy_for_particular_kz = []

for ky in ky_list:
    eigenvalues = eigens(ky, kz, phi)['Energy']
    energy_for_particular_kz.append(eigenvalues)
    
#np.savetxt("data_N200_ky_0_phi_pi_by2.txt",energy_for_particular_ky) 



#energ = np.loadtxt("data_N200_ky_0_phi_pi_by2.txt")   

energ =  np.array(energy_for_particular_kz)


plt.figure()
plt.plot(ky_list,energ[:,398:600],color='gray',alpha=0.3)
plt.plot(ky_list,energ[:,0:202],color='gray',alpha=0.3)
plt.xlabel('$k_y$',fontsize=16)
plt.ylabel('Energy',fontsize=16)
plt.title('Energy spectrum (before applying light),$\phi = 0, k_z =0$')
plt.ylim(-1,1)
plt.show()  

plt.savefig("plot_N200_ky_0_phi_pi_by2.pdf") 
 



    
    
               