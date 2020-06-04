# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:49:34 2018

@author: dominic

"""
import numpy as np

from tightbinding.bravaislattice import BravaisLattice
from tightbinding.tightbindinghamiltonianbuilder import TightBindingHamiltonianBuilder

def flatCheckerboardNNNNModel(W,H,t0,phase,t1,t2,t3,periodicBoundaryConditions=[]):
    """Returns the underlying lattice and built hamiltonian for a checkerboard
    model on a rectangular lattice of size W x H and nearest neighbor hopping t0
    with phase, next nearest neighbor hopping t1 or t2, and next next nerest negihbor hopping t3 
    The model is presented in : https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.106.236803
    """
    
    eX = np.array([1,0])
    eY = np.array([0,-1])
    
    eA = np.array([0,0])
    eB = np.array([0.5,0.5])
    
    lattice = BravaisLattice(np.matrix([eX,eY]),[W,H],np.array([eA,eB]),periodicBoundaryConditions)
    
    expphase = np.cos(phase) + 1j*np.sin(phase)
    
    hoppingDisplacementList = []
    hoppingMatrixList = []
    
    matrix00 = np.zeros((2,2),dtype=complex)
    matrix10 = np.zeros((2,2),dtype=complex)
    matrix01 = np.zeros((2,2),dtype=complex)
    matrix1m1 = np.zeros((2,2),dtype=complex)
    matrix11 = np.zeros((2,2),dtype=complex)
    
    hoppingDisplacementList.append([0,0])
    matrix00 = np.zeros((2,2),dtype=complex)
    matrix00[0][0] = 0
    matrix00[0][1] = t0*expphase
    matrix00[1][0] = t0*expphase.conj()
    matrix00[1][1] = 0
    hoppingMatrixList.append(matrix00)
    
    hoppingDisplacementList.append([1,0])
    matrix10[0][0] = t2
    matrix10[0][1] = 0
    matrix10[1][0] = t0*expphase
    matrix10[1][1] = t1
    hoppingMatrixList.append(matrix10)
    
    hoppingDisplacementList.append([0,1])
    matrix01[0][0] = t1
    matrix01[1][0] = t0*expphase.conj()
    matrix01[0][1] = 0
    matrix01[1][1] = t2
    hoppingMatrixList.append(matrix01)
    
    hoppingDisplacementList.append([1,-1])
    matrix1m1[0][0] = t3
    matrix1m1[0][1] = 0.0
    matrix1m1[1][0] = t0*expphase.conj()
    matrix1m1[1][1] = t3
    hoppingMatrixList.append(matrix1m1)
    
    hoppingDisplacementList.append([1,1])
    matrix11[0][0] = t3
    matrix11[0][1] = 0.0
    matrix11[1][0] = 0.0
    matrix11[1][1] = t3
    hoppingMatrixList.append(matrix11)
    
    builder = TightBindingHamiltonianBuilder(lattice,hoppingDisplacementList,hoppingMatrixList)
    
    return builder.generateSystem()
    