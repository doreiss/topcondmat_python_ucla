# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:36:05 2018

@author: dominic
"""

import numpy as np

from tightbinding.bravaislattice import oneDimensionalBipartiteLattice
from tightbinding.tightbindinghamiltonianbuilder import TightBindingHamiltonianBuilder

def Kitaev1dChain(L,mu,t,delta,periodicBoundaryConditions=[]):
    """Returns the underlying lattice and built hamiltonian for a kitaev
    superconducting chain in the ordinary fermion basis.
    """
    lattice = oneDimensionalBipartiteLattice(L,periodicBoundaryConditions)
    
    hoppingDisplacementList = []
    hoppingMatrixList = []
    
    matrix0 = np.zeros((2,2),dtype=complex)
    matrix1 = np.zeros((2,2),dtype=complex)
    
    hoppingDisplacementList.append([0])
    matrix0[0][0] = -mu/2.0
    matrix0[1][1] = mu/2.0
    hoppingMatrixList.append(matrix0)
    
    hoppingDisplacementList.append([1])
    matrix1[0][0] = -t/2.0
    matrix1[1][0] = -delta/2.0
    matrix1[0][1] = delta/2.0
    matrix1[1][1] = t/2.0
    hoppingMatrixList.append(matrix1)
    
    builder = TightBindingHamiltonianBuilder(lattice,hoppingDisplacementList,hoppingMatrixList)
    
    return builder.generateSystem()