#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 19:39:42 2024

@author: aneekphys
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cmath
import scipy as sp
from scipy.integrate import quad
plt.rcParams['text.usetex']=True

N = 200
m = 1
th = 1

del_eps = -1.0
bxy = 1
bz = 2
v = 1
A0 = 1
w = 3
eta = 2

iden = np.eye(N)

upper = np.zeros((N,N))

lower = np.zeros((N,N))

sig_x = np.array([[0,1],[1,0]])
sig_y = np.array([[0,-1j],[1j,0]])
sig_z = np.array([[1,0],[0,-1]])

def Ay(r, alpha,t):
    return (A0*(-r*np.sin(eta*w*t - alpha) + np.sin(w*t)))

def Az(r,alpha,t):
    return (A0*(r*np.cos(eta*w*t - alpha) + np.cos(w*t)))


for i in range(N-1):
    upper[i][i+1] = 1
    
for i in range(N-1):
    lower[i+1][i] = 1
    
# Without light Lattice Hamiltonian

# we are using open boubdary condition along Z direction------------

def Hd(kx,ky,Ay):
    
    #return (del_eps + 2*bxy*(2 - (1-kx**2/2) -(1-(ky+Ay)**2/2)) + 2*bz)*sig_z #+ ky*sig_x
    return (del_eps + 2*bxy*(2 - np.cos(kx)-np.cos(ky + Ay)) + 2*bz)*sig_z

def Hl(kx,ky,Az):
    return (-1j*v*0.5*sig_y - bz*sig_z)*np.exp(1j*Az)

def Hu(kx,ky,Az):
    return (1j*v*0.5*sig_y - bz*sig_z)*np.exp(-1j*Az)

def H_full(kx,ky,r,alpha):
    H0d = Hd(kx,ky,0)
    H0l = Hl(kx,ky,0)
    H0u = Hu(kx, ky, 0)
    H0 = np.kron(iden,H0d) + np.kron(lower,H0l) + np.kron(upper,H0u)
    
    ti, tf = 0, 2*np.pi/w
   
    Hp1d = np.zeros((2,2),dtype=np.complex128)
    Hm1d = np.zeros((2,2),dtype=np.complex128)
    Hp2d = np.zeros((2,2),dtype=np.complex128)
    Hm2d = np.zeros((2,2),dtype=np.complex128)
    Hp3d = np.zeros((2,2),dtype=np.complex128)
    Hm3d = np.zeros((2,2),dtype=np.complex128)
    Hp4d = np.zeros((2,2),dtype=np.complex128)
    Hm4d = np.zeros((2,2),dtype=np.complex128)
    
    Hp1u = np.zeros((2,2),dtype=np.complex128)
    Hm1u = np.zeros((2,2),dtype=np.complex128)
    Hp2u = np.zeros((2,2),dtype=np.complex128)
    Hm2u = np.zeros((2,2),dtype=np.complex128)
    Hp3u = np.zeros((2,2),dtype=np.complex128)
    Hm3u = np.zeros((2,2),dtype=np.complex128)
    Hp4u = np.zeros((2,2),dtype=np.complex128)
    Hm4u = np.zeros((2,2),dtype=np.complex128)
    
    Hp1l = np.zeros((2,2),dtype=np.complex128)
    Hm1l = np.zeros((2,2),dtype=np.complex128)
    Hp2l = np.zeros((2,2),dtype=np.complex128)
    Hm2l = np.zeros((2,2),dtype=np.complex128)
    Hp3l = np.zeros((2,2),dtype=np.complex128)
    Hm3l = np.zeros((2,2),dtype=np.complex128)
    Hp4l = np.zeros((2,2),dtype=np.complex128)
    Hm4l = np.zeros((2,2),dtype=np.complex128)
    
    for i in range(2):
        for j in range(2):
            
    
            Hp1d[i,j], _ = (w/(2*np.pi) *np.array(quad(lambda t : np.real(np.exp(-1j* (1) * w*t)*Hd(kx,ky,Ay(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (1) * w*t)*Hd(kx,ky,Ay(r, alpha, t))[i,j]),ti,tf)))
            
            Hm1d[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-1) * w*t)*Hd(kx,ky,Ay(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-1) * w*t)*Hd(kx,ky,Ay(r, alpha, t))[i,j]),ti,tf)))
            
            Hp2d[i,j], _ = (w/(2*np.pi) *np.array(quad(lambda t : np.real(np.exp(-1j* (2) * w*t)*Hd(kx,ky,Ay(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (2) * w*t)*Hd(kx,ky,Ay(r, alpha, t))[i,j]),ti,tf)))
            
            Hm2d[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-2) * w*t)*Hd(kx,ky,Ay(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-2) * w*t)*Hd(kx,ky,Ay(r, alpha, t))[i,j]),ti,tf)))
            
            Hp3d[i,j], _ = (w/(2*np.pi) *np.array(quad(lambda t : np.real(np.exp(-1j* (3) * w*t)*Hd(kx,ky,Ay(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (3) * w*t)*Hd(kx,ky,Ay(r, alpha, t))[i,j]),ti,tf)))
            
            Hm3d[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-3) * w*t)*Hd(kx,ky,Ay(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-3) * w*t)*Hd(kx,ky,Ay(r, alpha, t))[i,j]),ti,tf)))
            
            Hp4d[i,j], _ = (w/(2*np.pi) *np.array(quad(lambda t : np.real(np.exp(-1j* (4) * w*t)*Hd(kx,ky,Ay(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (4) * w*t)*Hd(kx,ky,Ay(r, alpha, t))[i,j]),ti,tf)))
            
            Hm4d[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-4) * w*t)*Hd(kx,ky,Ay(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-4) * w*t)*Hd(kx,ky,Ay(r, alpha, t))[i,j]),ti,tf)))
            
            
            
            Hp1u[i,j], _ = (w/(2*np.pi) *np.array(quad(lambda t : np.real(np.exp(-1j* (1) * w*t)*Hu(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (1) * w*t)*Hu(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))
            
            Hm1u[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-1) * w*t)*Hu(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-1) * w*t)*Hu(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))
            
            Hp2u[i,j], _ = (w/(2*np.pi) *np.array(quad(lambda t : np.real(np.exp(-1j* (2) * w*t)*Hu(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (2) * w*t)*Hu(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))
            
            Hm2u[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-2) * w*t)*Hu(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-2) * w*t)*Hu(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))
            
            Hp3u[i,j], _ = (w/(2*np.pi) *np.array(quad(lambda t : np.real(np.exp(-1j* (3) * w*t)*Hu(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (3) * w*t)*Hu(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))
            
            Hm3u[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-3) * w*t)*Hu(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-3) * w*t)*Hu(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))
            
            Hp4u[i,j], _ = (w/(2*np.pi) *np.array(quad(lambda t : np.real(np.exp(-1j* (4) * w*t)*Hu(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (4) * w*t)*Hu(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))
            
            Hm4u[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-4) * w*t)*Hu(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-4) * w*t)*Hu(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))
            
            
            Hp1l[i,j], _ = (w/(2*np.pi) *np.array(quad(lambda t : np.real(np.exp(-1j* (1) * w*t)*Hl(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (1) * w*t)*Hl(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))
            
            Hm1l[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-1) * w*t)*Hl(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-1) * w*t)*Hl(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))
            
            Hp2l[i,j], _ = (w/(2*np.pi) *np.array(quad(lambda t : np.real(np.exp(-1j* (2) * w*t)*Hl(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (2) * w*t)*Hl(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))
            
            Hm2l[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-2) * w*t)*Hl(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-2) * w*t)*Hl(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))
            
            Hp3l[i,j], _ = (w/(2*np.pi) *np.array(quad(lambda t : np.real(np.exp(-1j* (3) * w*t)*Hl(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (3) * w*t)*Hl(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))
            
            Hm3l[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-3) * w*t)*Hl(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-3) * w*t)*Hl(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))
            
            Hp4l[i,j], _ = (w/(2*np.pi) *np.array(quad(lambda t : np.real(np.exp(-1j* (4) * w*t)*Hl(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (4) * w*t)*Hl(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))
            
            Hm4l[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-4) * w*t)*Hl(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-4) * w*t)*Hl(kx,ky,Az(r, alpha, t))[i,j]),ti,tf)))
            
            
    Hp1 = np.kron(iden,Hp1d) + np.kron(lower,Hp1l) + np.kron(upper,Hp1u)
    Hm1 = np.kron(iden,Hm1d) + np.kron(lower,Hm1l) + np.kron(upper,Hm1u)
    Hp2 = np.kron(iden,Hp2d) + np.kron(lower,Hp2l) + np.kron(upper,Hp2u)
    Hm2 = np.kron(iden,Hm2d) + np.kron(lower,Hm2l) + np.kron(upper,Hm2u)
    Hp3 = np.kron(iden,Hp3d) + np.kron(lower,Hp3l) + np.kron(upper,Hp3u)
    Hm3 = np.kron(iden,Hm3d) + np.kron(lower,Hm3l) + np.kron(upper,Hm3u)
    Hp4 = np.kron(iden,Hp4d) + np.kron(lower,Hp4l) + np.kron(upper,Hp4u)
    Hm4 = np.kron(iden,Hm4d) + np.kron(lower,Hm4l) + np.kron(upper,Hm4u)
    
    #print("\n Hp1d",Hp1d,"\n Hp1u",Hp1u)
    
    #print("\n Hp1", Hp1, "\n Hm1", Hm1)
    
    #print("\n comm.", Hp1 @ Hm1 - Hm1 @ Hp1)
    
    H1 = (1/(1*w))*((Hp1@Hm1 - Hm1@Hp1) + 0.5*((H0@Hp1-Hp1@H0)-(H0@Hm1 -Hm1@H0 )))
    H2 = (1/(2*w))*((Hp2@Hm2 - Hm2@Hp2) + 0.5*((H0@Hp2-Hp2@H0)-(H0@Hm2 -Hm2@H0 )))
    H3 = (1/(3*w))*((Hp3@Hm3 - Hm3@Hp3) + 0.5*((H0@Hp3-Hp3@H0)-(H0@Hm3 -Hm3@H0 )))
    H4 = (1/(4*w))*((Hp4@Hm4 - Hm4@Hp4) + 0.5*((H0@Hp4-Hp4@H0)-(H0@Hm4 -Hm4@H0 )))
    
    #print("\n H1",H1)
    
    return (H0 + H1 + H2 + H3 + H4), H0



def eigens(kx,ky,r,alpha):
    E,V = np.linalg.eigh(H_full(kx, ky,r,alpha)[0])
    
    Dict ={}
    Dict['Energy'] = E
    Dict['Eigenvec'] = V
    
    return Dict

kx_list = np.linspace(-np.pi,np.pi,100)
ky_list = np.linspace(-np.pi,np.pi,100)

#----------------------------------------------------------------------------------
r = 1

alpha = 0


kx = 0



energy_for_particular_kx = []

for ky in ky_list:
    eigenvalues = eigens(kx, ky,r,alpha)['Energy']
    energy_for_particular_kx.append(eigenvalues)
    
    
    
ky = 0


energy_for_particular_ky = []

for kx in kx_list:
    eigenvalues = eigens(kx, ky,r, alpha)['Energy']
    energy_for_particular_ky.append(eigenvalues)

    
energ1 =  np.array(energy_for_particular_kx)

energ2 =  np.array(energy_for_particular_ky)


os.chdir('/Users/aneekphys/Library/CloudStorage/OneDrive-IndianInstituteofScience/bcl light/codes/python/line_node_semimetal/final_dataplots')

np.savetxt('kx_list.txt',kx_list)
np.savetxt('ky_list.txt',ky_list)
np.savetxt('eta_2_alpha_0_r_1_bxy_neq_bz_energy_for_particular_ky.txt', energ2)
np.savetxt('eta_2_alpha_0_r_1_bxy_neq_bz_energy_for_particular_kx.txt', energ1)



kx=0
ky=0




plt.figure()
plt.plot(ky_list,energ1,color='gray',alpha=0.3)
plt.xlabel('$k_y$',fontsize=22)
plt.ylabel('$E$',fontsize=22)
ax = plt.gca()
#plt.title(r'Energy spectrum with BCL, $\Delta\epsilon$={} $k_x$={:.2f} r={:.2f} $\alpha$={:.2f}'.format(del_eps,kx,r,alpha))
plt.ylim(-2,2)

# Set the border color and width
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Set the border color
    spine.set_linewidth(1.5)      # Set the border width

plt.xticks(fontsize=22)
plt.yticks([-2.0,-1.0,0.0,1.0,2.0],fontsize=22)


plt.tight_layout(pad=4.0)
plt.show()

plt.savefig('/Users/aneekphys/Documents/BCL plots/num_line-node_eta_2_alpha_0_r_1_bxy_neq_bz_kx_fixed.pdf',bbox_inches='tight')




plt.figure()
plt.plot(kx_list,energ2,color='gray',alpha=0.3)
plt.xlabel('$k_x$',fontsize=22)
plt.ylabel('$E$',fontsize=22)
ax = plt.gca()
#plt.title(r'Energy spectrum with BCL,$\Delta\epsilon$={} $k_y$={:.2f} r={:.2f} $\alpha$={:.2f}'.format(del_eps,ky,r,alpha))
plt.ylim(-2,2)

# Set the border color and width
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Set the border color
    spine.set_linewidth(1.5)      # Set the border width

plt.xticks(fontsize=22)
plt.yticks([-2.0,-1.0,0.0,1.0,2.0],fontsize=22)


plt.tight_layout(pad=4.0)
plt.show()  

plt.savefig('/Users/aneekphys/Documents/BCL plots/num_line-node_eta_2_alpha_0_r_1_bxy_neq_bz_ky_fixed.pdf',bbox_inches='tight')



        
'''

H, H0 = H_full(1, 1, r, alpha)

print("\n\n",H-H0)

 '''    
           
