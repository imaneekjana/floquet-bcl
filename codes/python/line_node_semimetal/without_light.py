#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 18:59:08 2024

@author: aneekphys
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath

N = 200
m = 1
th = 1

del_eps = -2
bxy = 1
bz = 1
v = 1

iden = np.eye(N)

upper = np.zeros((N,N))

lower = np.zeros((N,N))

sig_x = np.array([[0,1],[1,0]])
sig_y = np.array([[0,-1j],[1j,0]])
sig_z = np.array([[1,0],[0,-1]])


for i in range(N-1):
    upper[i][i+1] = 1
    
for i in range(N-1):
    lower[i+1][i] = 1
    
# Without light Lattice Hamiltonian

# we are using open boubdary condition along Z direction------------

def Hd(kx,ky):
    return (del_eps + 2*bxy*(2 - np.cos(kx)-np.cos(ky)) + 2*bz)*sig_z

def Hl(kx,ky):
    return (-1j*v*0.5*sig_y - bz*sig_z)

def Hu(kx,ky):
    return (1j*v*0.5*sig_y - bz*sig_z)

def H_full(kx,ky):
    return (np.kron(iden,Hd(kx,ky)) + np.kron(lower,Hl(kx,ky)) + np.kron(upper,Hu(kx,ky)))

def eigens(kx,ky):
    E,V = np.linalg.eigh(H_full(kx, ky))
    
    Dict ={}
    Dict['Energy'] = E
    Dict['Eigenvec'] = V
    
    return Dict

kx_list = np.linspace(-np.pi,np.pi,N)
ky_list = np.linspace(-np.pi,np.pi,N)



kx = 0


energy_for_particular_kx = []

for ky in ky_list:
    eigenvalues = eigens(kx, ky)['Energy']
    energy_for_particular_kx.append(eigenvalues)
    
    
    
ky = 0


energy_for_particular_ky = []

for kx in kx_list:
    eigenvalues = eigens(kx, ky)['Energy']
    energy_for_particular_ky.append(eigenvalues)    
    
#np.savetxt("data_N200_ky_0_phi_pi_by2.txt",energy_for_particular_ky) 



#energ = np.loadtxt("data_N200_ky_0_phi_pi_by2.txt")   

energ1 =  np.array(energy_for_particular_kx)

energ2 =  np.array(energy_for_particular_ky)


plt.figure()
plt.plot(ky_list,energ1,color='gray',alpha=0.3)
plt.xlabel('$k_y$',fontsize=16)
plt.ylabel('Energy',fontsize=16)
plt.title('Energy spectrum (before applying light),$ k_x =0$$')
plt.ylim(-2,2)
plt.show()

plt.figure()
plt.plot(kx_list,energ2,color='gray',alpha=0.3)
plt.xlabel('$k_x$',fontsize=16)
plt.ylabel('Energy',fontsize=16)
plt.title('Energy spectrum (before applying light), $k_y =0$$')
plt.ylim(-2,2)
plt.show()  

#plt.savefig("plot_N200_ky_0_phi_pi_by2.pdf") 
