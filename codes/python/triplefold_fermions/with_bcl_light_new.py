#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:28:10 2024

@author: aneekphys
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cmath
import scipy as sp
from scipy.integrate import quad

#os.chdir('bcl_data/open_boundary_along_x_direction/')
from matplotlib import rc


plt.rcParams['text.usetex']=True


#rc('font', family='serif')

N = 200
m = 1
th = 1
w = 3
eta = 2
No = 200

phi = np.pi/6

alpha = 0

r = np.sqrt(eta) #np.sqrt(eta) #np.sqrt(eta) 

A0 = 0.5 #1/np.sqrt(1+r**2) #1/r if r>1 else 1   #keep maximum amplitude to be 1

kyshift = 0.000 #A0**3*np.tan(alpha)*(eta*(1/np.cos(alpha))**2-1)/(4*w**2 + A0**2 * np.tan(alpha)**2)


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



def H_full_eff(ky,kz,phi,r,alpha):
    
    ## The main diagonal Fourier components
    
    ti, tf = 0, 2*np.pi/w
    H0d = Hd(ky,kz,phi,Ay=0)
    #H0 = np.zeros((3,3),dtype=np.complex128)
    Hp1d = np.zeros((3,3),dtype=np.complex128)
    Hm1d = np.zeros((3,3),dtype=np.complex128)
    Hp_etad = np.zeros((3,3),dtype=np.complex128)
    Hm_etad = np.zeros((3,3),dtype=np.complex128)
    
    for i in range(3):
        for j in range(3):
            
            #H0[i,j], _ = (w/(2*np.pi) *np.array(quad(lambda t : np.real(np.exp(-1j* (0) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (0) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))
            
            Hp1d[i,j], _ = (w/(2*np.pi) *np.array(quad(lambda t : np.real(np.exp(-1j* (1) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (1) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))
            
            Hm1d[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-1) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-1) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))
            
            Hp_etad[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (eta) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (eta) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))
            
            Hm_etad[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-eta) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-eta) * w*t)*Hd(ky,kz,phi,Ay(r, alpha, t))[i,j]),ti,tf)))
            
    
    ## The upper diagonal Fourier 
    
    ti, tf = 0, 2*np.pi/w
    H0u = H_off_d(ky,kz,phi,Ax=0)['Hu']
    Hp1u = np.zeros((3,3),dtype=np.complex128)
    Hm1u = np.zeros((3,3),dtype=np.complex128)
    Hp_etau = np.zeros((3,3),dtype=np.complex128)
    Hm_etau = np.zeros((3,3),dtype=np.complex128)
    
    for i in range(3):
        for j in range(3):
            
            #H0[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (0) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (0) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))
            
            Hp1u[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (1) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (1) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))
            
            Hm1u[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-1) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-1) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))
            
            Hp_etau[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (eta) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (eta) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))
            
            Hm_etau[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-eta) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-eta) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))
            
    
    
    ## The lower diagonal
    
    ti, tf = 0, 2*np.pi/w
    H0l = H_off_d(ky,kz,phi,Ax=0)['Hl']
    Hp1l = np.zeros((3,3),dtype=np.complex128)
    Hm1l = np.zeros((3,3),dtype=np.complex128)
    Hp_etal = np.zeros((3,3),dtype=np.complex128)
    Hm_etal = np.zeros((3,3),dtype=np.complex128)
    
    for i in range(3):
        for j in range(3):
            
            #H0[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (0) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (0) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hu'][i,j]),ti,tf)))
            
            Hp1l[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (1) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hl'][i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (1) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hl'][i,j]),ti,tf)))
            
            Hm1l[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-1) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hl'][i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-1) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hl'][i,j]),ti,tf)))
            
            Hp_etal[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (eta) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hl'][i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (eta) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hl'][i,j]),ti,tf)))
            
            Hm_etal[i,j], _ = (w/(2*np.pi) * np.array(quad(lambda t : np.real(np.exp(-1j* (-eta) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hl'][i,j]),ti,tf)))+1.0j*(w/(2*np.pi) * np.array(quad(lambda t : np.imag(np.exp(-1j* (-eta) * w*t)*H_off_d(ky,kz,phi,Ax(r, alpha, t))['Hl'][i,j]),ti,tf)))
    
     
    
    
    H0 = np.kron(iden,H0d)+np.kron(upper,H0u)+np.kron(lower,H0l)
    Hp1 = np.kron(iden,Hp1d)+np.kron(upper,Hp1u)+np.kron(lower,Hp1l)
    Hm1 = np.kron(iden,Hm1d)+np.kron(upper,Hm1u)+np.kron(lower,Hm1l)
    Hp_eta = np.kron(iden,Hp_etad)+np.kron(upper,Hp_etau)+np.kron(lower,Hp_etal)
    Hm_eta = np.kron(iden,Hm_etad)+np.kron(upper,Hm_etau)+np.kron(lower,Hm_etal)
    
    H1 = (1/w)*(Hp1 @ Hm1 - Hm1 @ Hp1) + (1/(eta*w))*(Hp_eta @ Hm_eta - Hm_eta @ Hp_eta)
    H2 = (1/(2*w))*((H0 @ Hp1 - Hp1 @ H0)-(H0 @ Hm1 - Hm1 @ H0)) 
    H3 = (1/(2*eta*w))*((H0 @ Hp_eta - Hp_eta @ H0)-(H0 @ Hm_eta - Hm_eta @ H0)) 
    
    return (H0 + H1 + H2 + H3)
    
    


    
def eigens(ky,kz,phi,r,alpha):
    E,V = np.linalg.eigh(H_full_eff(ky, kz, phi,r,alpha))
    
    Dict ={}
    Dict['Energy'] = E
    Dict['Eigenvec'] = V
    
    return Dict


ky_list = np.linspace(-np.pi,np.pi,No)
kz_list = np.linspace(-np.pi,np.pi,No)

#####----------- Effect of BCL -----------------



## particular ky

ky = 0+kyshift # for band-structure at ky=0 and as a function of kz


energy_for_particular_ky = []

for kz in kz_list:
    eigenvalues = eigens(ky, kz, phi,r,alpha)['Energy']
    energy_for_particular_ky.append(eigenvalues)
    
#np.savetxt("data_ky_{:.3f}_A_{:.2f}_phi_{:.2f}_r_{:.2f}_eta_{}_alpha_{:.2f}.txt".format(ky,A0,phi,r,eta,alpha),energy_for_particular_ky)  

energ = energy_for_particular_ky

np.savetxt('/Users/aneekphys/Library/CloudStorage/OneDrive-IndianInstituteofScience/bcl light/codes/python/triplefold_fermions/final_data_plots/kz_list.txt',kz_list)
np.savetxt('/Users/aneekphys/Library/CloudStorage/OneDrive-IndianInstituteofScience/bcl light/codes/python/triplefold_fermions/final_data_plots/band_eta_2_r_1.41_fixed_ky_phi_piby6_alpha_0.pdf.txt',energ)




#energ = np.loadtxt("data_ky_{:.3f}_A_{:.2f}_phi_{:.2f}_r_{:.2f}_eta_{}_alpha_{:.2f}.txt".format(ky,A0,phi,r,eta,alpha))    
plt.rcParams['text.usetex']=True
plt.figure()



plt.plot(kz_list,energ,color='gray',alpha=0.4)

plt.xlabel('$k_z$',fontsize=22)
plt.ylabel('$E$',fontsize=22)

ax = plt.gca()

# Set the border color and width
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # Set the border color
    spine.set_linewidth(1.5)      # Set the border width

#plt.title(r'$\phi = {:.2f}, k_y ={:.3f}, A0={:.2f}$, $r = {:.2f}, \eta = {},\alpha = {:.2f}$'.format(phi,ky,A0,r,eta,alpha))
plt.ylim(-1,1)
plt.xlim(-np.pi,np.pi)

plt.axvline(x=-np.pi/2)
#plt.tick_params(axis='both', which='major', labelsize=18)

plt.xticks(fontsize=22)
plt.yticks([-1,-0.5,0.0,0.5,1.0],fontsize=22)

plt.tight_layout(pad=4.0)

plt.show()  

#plt.savefig("plot_ky_{:.3f}_A_{:.2f}_phi_{:.2f}_r_{:.2f}_eta_{}_alpha_{:.2f}.pdf".format(ky,A0,phi,r,eta,alpha)) 
 
plt.savefig('/Users/aneekphys/Documents/BCL plots/num_bcl_eta_2_r_sqrt_2_fixed_ky_phi_pi_by_6_alpha_0.pdf',bbox_inches='tight')


'''

## particular kz

kz = 3*np.pi/4 # for band-structure at kz=kz and as a function of ky


energy_for_particular_kz = []

for ky in ky_list:
    eigenvalues = eigens(ky, kz, phi,r,alpha)['Energy']
    energy_for_particular_kz.append(eigenvalues)
    
np.savetxt("data_kz_{:.3f}_A_{:.2f}_phi_{:.2f}_r_{:.2f}_eta_{}_alpha_{:.2f}.txt".format(kz,A0,phi,r,eta,alpha),energy_for_particular_kz)  

#energ = energy_for_particular_ky


energ = np.loadtxt("data_kz_{:.3f}_A_{:.2f}_phi_{:.2f}_r_{:.2f}_eta_{}_alpha_{:.2f}.txt".format(kz,A0,phi,r,eta,alpha))    

plt.figure()
plt.plot(ky_list,energ[:,0:203],color='gray',alpha=0.3)
plt.plot(ky_list,energ[:,397:600],color='gray',alpha=0.3)
plt.xlabel('$k_y$',fontsize=16)
plt.ylabel('Energy',fontsize=16)
plt.title(r'$\phi = {:.2f}, k_z ={:.3f}, A0={:.2f}$, $r = {:.2f}, \eta = {},\alpha = {:.2f}$'.format(phi,kz,A0,r,eta,alpha))
plt.ylim(-1,1)
plt.xlim(-np.pi,np.pi)
plt.axvline(x=-np.pi/2)
plt.show()  

plt.savefig("plot_kz_{:.3f}_A_{:.2f}_phi_{:.2f}_r_{:.2f}_eta_{}_alpha_{:.2f}.pdf".format(kz,A0,phi,r,eta,alpha)) 
 
'''











    

    
    
               