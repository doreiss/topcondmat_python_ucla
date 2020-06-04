# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:57:41 2018

@author: dominic
"""

import scipy
import numpy as np
from scipy.linalg import expm

from mathtools.liealgebra import SpecialUnitaryAlgebra

def normalize(v):
    """Normalizes an np array
    """
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v/norm

def randomUnitVector(n):
    """Generates a random unit vector in n dimensions
    """
    v = np.random.normal(0,1.0,n)
    return normalize(v)

class SmallUnitaryBuilder:
    """Builds random unitary operators a 'small' distance from the identity
    """
    def __init__(self,n):
        self.n = n
        self.algebra = SpecialUnitaryAlgebra(self.n)
    
    def randomSmallUnitary(self,t):
        v = randomUnitVector(self.algebra.dimension)
        matrix = self.algebra.matrixRepresentation(v)
        return expm(-t*1j*matrix)
    
def randomHaarUnitary(n):
    """A Haar distributed random n x n unitary matrix
    """
    z = (scipy.randn(n,n) + 1j*scipy.randn(n,n))/scipy.sqrt(2.0)
    q,r = np.linalg.qr(z)
    d = scipy.diagonal(r)
    ph = d/scipy.absolute(d)
    q = scipy.multiply(q,ph,q)
    return q

