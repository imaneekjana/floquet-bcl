#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:28:10 2024

@author: aneekphys
"""
import numpy as np
import matplotlib.pyplot as plt
import cmath
import scipy as sp
from scipy.integrate import quad




N = 200
m = 1
th = 1
A0 = 5
w = 3
eta = 3

iden = np.eye(N)

upper = np.zeros((N,N))

lower = np.zeros((N,N))

for i in range(N-1):
    upper[i][i+1] = 1
    
for i in range(N-1):
    lower[i+1][i] = 1
    
#--------BCL--------
def Ax(r,alpha,t):
    return A0*(r*np.cos(alpha - eta*w*t)+ np.cos(w*t))

def Ay(r,alpha,t):
    return A0*(r*np.sin(alpha - eta*w*t)+ np.sin(w*t))
     
    
# With light Lattice Hamiltonian

def Hd(ky,kz,phi,Ay):
    Hd_p = np.zeros((3,3),dtype=np.complex128)
    Hd_p[0,2] = np.exp(-1j *phi)* th * np.sin(ky + Ay)
    Hd_p[1,2] = np.exp(1j*phi)* m *(np.cos(kz)+ np.cos(ky + Ay) -2)
    Hd_p[2,0] = np.exp(1j* phi) * th * np.sin(ky + Ay)
    Hd_p[2,1] = np.exp(-1j*phi)*(np.cos(kz) + np.cos(ky + Ay) -2)
    return Hd_p

def H_off_d(ky,kz,phi,Ax):
    Hl_p = np.zeros((3,3),dtype=np.complex128)
    Hl_p[0,1] = np.exp(1j*phi) * th * np.exp(1j*Ax)/(2*1j)
    Hl_p[1,0] = np.exp(-1j*phi)*th*np.exp(1j*Ax) /(2*1j)
    Hl_p[1,2] = np.exp(1j*phi)*np.exp(1j*Ax) * m/2
    Hl_p[2,1] = np.exp(-1j*phi)*np.exp(1j*Ax) *m/2
    Hu_p = np.conjugate(Hl_p.T)
    
    Dict = {}
    
    Dict['Hl'] = Hl_p
    Dict['Hu'] = Hu_p
    
    return Dict



def Hd_eff(ky,kz,phi,r,alpha):
    
    ti, tf = 0, 2*np.pi/w
    H0 = Hd(ky,kz,phi,Ay=0)
    #H0 = np.zeros((3,3),dtype=np.complex128)
    Hp1 = np.zeros((3,3),dtype=np.complex128)
    Hm1 = np.zeros((3,3),dtype=np.complex128)
    Hp_eta = np.zeros((3,3),dtype=np.complex128)
    Hm_eta = np.zeros((3,3),dtype=np.complex128)
    
    for i in range(3):
        for j in range(3):
            
            #H0[i,j], _ = (w/(2*np.pi) *np.array(quad(lambda t : np.real(np.exp(-1j* (0) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (0) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))
            
            Hp1[i,j], _ = (w/(2*np.pi) *np.array(quad(lambda t : np.real(np.exp(-1j* (1) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (1) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))
            
            Hm1[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-1) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-1) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))
            
            Hp_eta[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (eta) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (eta) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))
            
            Hm_eta[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-eta) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-eta) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))
            
    
    H1 = (1/w)*(Hp1 @ Hm1 - Hm1 @ Hp1) + (1/(eta*w))*(Hp_eta @ Hm_eta - Hm_eta @ Hp_eta)
    H2 = (1/(2*w))*((H0 @ Hp1 - Hp1 @ H0)-(H0 @ Hm1 - Hm1 @ H0)) 
    H3 = (1/(2*eta*w))*((H0 @ Hp_eta - Hp_eta @ H0)-(H0 @ Hm_eta - Hm_eta @ H0)) 
    
    return (H0+H1+H2+H3)


def Hoff_d_eff(ky,kz,phi,r,alpha):
    
    ti, tf = 0, 2*np.pi/w
    H0 = H_off_d(ky,kz,phi,Ax=0)['Hu']
    #H0 = np.zeros((3,3),dtype=np.complex128)
    Hp1 = np.zeros((3,3),dtype=np.complex128)
    Hm1 = np.zeros((3,3),dtype=np.complex128)
    Hp_eta = np.zeros((3,3),dtype=np.complex128)
    Hm_eta = np.zeros((3,3),dtype=np.complex128)
    
    for i in range(3):
        for j in range(3):
            
            #H0[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (0) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (0) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))
            
            Hp1[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (1) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (1) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))
            
            Hm1[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-1) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-1) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))
            
            Hp_eta[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (eta) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (eta) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))
            
            Hm_eta[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-eta) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-eta) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))
            
    
    H1 = (1/w)*(Hp1 @ Hm1 - Hm1 @ Hp1) + (1/(eta*w))*(Hp_eta @ Hm_eta - Hm_eta @ Hp_eta)
    H2 = (1/(2*w))*((H0 @ Hp1 - Hp1 @ H0)-(H0 @ Hm1 - Hm1 @ H0)) 
    H3 = (1/(2*eta*w))*((H0 @ Hp_eta - Hp_eta @ H0)-(H0 @ Hm_eta - Hm_eta @ H0)) 
    
    Hueff = (H0+H1+H2+H3)
    Hleff = np.conjugate(Hueff.T)
    Dict = {}
    Dict['Hl'] = Hleff
    Dict['Hu'] = Hueff
    
    return Dict
     
    



    
def H_full(ky,kz,phi,r,alpha):
    H = np.kron(iden,Hd_eff(ky,kz,phi,r,alpha)) + np.kron(upper,Hoff_d_eff(ky,kz,phi,r,alpha)['Hu']) + np.kron(lower,Hoff_d_eff(ky,kz,phi,r,alpha)['Hl'])
    return H
    
def eigens(ky,kz,phi,r,alpha):
    E,V = np.linalg.eigh(H_full(ky, kz, phi,r,alpha))
    
    Dict ={}
    Dict['Energy'] = E
    Dict['Eigenvec'] = V
    
    return Dict


ky_list = np.linspace(-np.pi,np.pi,N)
kz_list = np.linspace(-np.pi,np.pi,N)

#####----------- Effect of BCL -----------------



phi = np.pi/6
ky = 0
r = eta
alpha =0


energy_for_particular_ky = []

for kz in kz_list:
    eigenvalues = eigens(ky, kz, phi,r,alpha)['Energy']
    energy_for_particular_ky.append(eigenvalues)
    
#np.savetxt("bcl_data/open_boundary_along_x_direction/data_ky_0_A_{}_phi_{}_r_{}_eta_{}_alpha_{}.txt".format(A0,phi,r,eta,alpha),energy_for_particular_ky)  

energ = energy_for_particular_ky

#energ = np.loadtxt("bcl_data/open_boundary_along_x_direction/data_ky_0_A_{}_phi_{}_r_{}_eta_{}_alpha_{}.txt".format(A0,phi,r,eta,alpha))    

plt.figure()
plt.plot(kz_list,energ)
plt.xlabel('$k_z$',fontsize=16)
plt.ylabel('Energy',fontsize=16)
plt.title('Energy spectrum with BCL,$\phi = {}, k_y =0, A0={}$, $r = {}, \eta = {},\alpha = {}$'.format(phi,A0,r,eta,alpha))
plt.ylim(-1,1)
plt.xlim(-np.pi,np.pi)
plt.axvline(x=-np.pi/2)
plt.show()  

#plt.savefig("bcl_data/open_boundary_along_x_direction/plot_ky_0_A_{}_phi_{}_r_{}_eta_{}_alpha_{}.pdf".format(A0,phi,r,eta,alpha)) 
 


    

    
    
               