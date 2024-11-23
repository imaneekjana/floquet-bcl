#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:25:56 2024

@author: aneekphys
"""

import numpy as np

import matplotlib.pyplot as plt
import cmath

sig_y = np.array([[0, -1j],[1j,0]])

sig_z = np.array([[1,0],[0,-1]])

sig_x = np.array([[0,1],[1,0]])


bxy = 1
v = 1
del_epsil = -0.2
bz = 1.1

def H(kx,ky,kz):
    return (v*kz*sig_y + (del_epsil + bxy * (kx**2 + ky**2) + bz*kz**2)*sig_z)

def DetM(kx,ky,E):
    a = -bz**2
    b = 0
    c = -2*bxy*bz*(kx**2 + ky**2) - 2*bz*del_epsil
    d =0
    e = -bxy**2*(kx**2 + ky**2)**2 - 2 * bxy* del_epsil*(kx**2 + ky**2) - (del_epsil)**2 + E**2
    coefficients = [a, b, c, d, e]
    kz_roots = np.roots(coefficients)
    negative_imag_roots = []
    
    for r in kz_roots:
        if np.imag(r) < 0:
            negative_imag_roots.append(r)
            
            
    for kz in negative_imag_roots:
        Hamil = H(kx,ky,kz)
        E_p, V = np.linalg.eig(Hamil)
        Kz_interest = []
        tol = 10**(-3)
        
        for i in E_p:
            if abs(E-i) <= tol:
                Kz_interest.append(kz)
                
    return Kz_interest , negative_imag_roots           
        
    
    
    
    
    
    
    
