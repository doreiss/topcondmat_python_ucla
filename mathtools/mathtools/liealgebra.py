# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 09:42:24 2018

@author: dominic
"""
import math
import numpy as np

class SpecialUnitaryAlgebra:
    def __init__(self,n):
        self.n = n
        self.dimension = (self.n-1)*(self.n+1)
        self.basis = []
        self.__generateBasis()
        self.basis = np.array(self.basis)
    
    def dot(self,matrixRepresentation1,matrixRepresentation2):
       return np.trace(matrixRepresentation1 @ matrixRepresentation2)
    
    def matrixRepresentation(self,vectorRepresentation):
        return np.tensordot(vectorRepresentation,self.basis,axes=[0,0])
    
    def vectorRepresentation(self,matrixRepresentation):
        return np.array([self.dot(matrixRepresentation,basisElem) for basisElem in self.basis])
    
    def __generateBasis(self):
        invsqrt2 = 1/math.sqrt(2.0)
        
        for i in range(self.n):
            for j in range(i):
                A = np.zeros((self.n,self.n),dtype=complex)
                B = np.zeros((self.n,self.n),dtype=complex)
                A[i,j] = invsqrt2
                A[j,i] = invsqrt2
                B[i,j] = invsqrt2*1j
                B[j,i] = -invsqrt2*1j
                
                self.basis.append(A)
                self.basis.append(B)
                
        for i in range(1,self.n):
            invsqrti = 1/math.sqrt(i+i*i)
            C = np.zeros((self.n,self.n), dtype=complex)      
            C[i,i] = -invsqrti * i
            for j in range(i):
                C[j,j] = invsqrti
            self.basis.append(C)
    