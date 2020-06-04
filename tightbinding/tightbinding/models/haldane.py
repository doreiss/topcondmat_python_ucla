# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 13:48:16 2018

@author: dominic
"""
import numpy as np

from tightbinding.bravaislattice import honeycombLattice
from tightbinding.tightbindinghamiltonianbuilder import TightBindingHamiltonianBuilder

def haldaneHoneycombSystem(W,H,t1,t2,M,periodicBoundaryConditions=[]):
    """Returns the underlying lattice and built hamiltonian for a haldane 
    model on a honeycomb lattice of size W x H and nearest neighbor hopping t1, 
    next nearest neighbor hopping t2 (or t2.conj() as appropraite), and 
    sublattice potential M 
    """
    lattice = honeycombLattice(W,H,periodicBoundaryConditions)
    
    t2conj = np.conj(t2)
    
    hoppingDisplacementList = []
    hoppingMatrixList = []
    
    matrix00 = np.zeros((2,2),dtype=complex)
    matrix10 = np.zeros((2,2),dtype=complex)
    matrix01 = np.zeros((2,2),dtype=complex)
    matrix1m1 = np.zeros((2,2),dtype=complex)
    
    hoppingDisplacementList.append(np.array([0,0],dtype=int))
    matrix00[0][0] = M
    matrix00[0][1] = t1
    matrix00[1][0] = t1
    matrix00[1][1] = -M
    hoppingMatrixList.append(matrix00)
    
    hoppingDisplacementList.append(np.array([1,0],dtype=int))
    matrix10[0][0] = t2
    matrix10[0][1] = t1
    matrix10[1][0] = 0.0
    matrix10[1][1] = t2conj
    hoppingMatrixList.append(matrix10)
    
    hoppingDisplacementList.append(np.array([0,1],dtype=int))
    matrix01[0][0] = t2conj
    matrix01[0][1] = t1
    matrix01[1][0] = 0.0
    matrix01[1][1] = t2
    hoppingMatrixList.append(matrix01)
    
    hoppingDisplacementList.append(np.array([1,-1],dtype=int))
    matrix1m1[0][0] = t2conj
    matrix1m1[0][1] = 0.0
    matrix1m1[1][0] = 0.0
    matrix1m1[1][1] = t2
    hoppingMatrixList.append(matrix1m1)
    
    builder = TightBindingHamiltonianBuilder(lattice,hoppingDisplacementList,hoppingMatrixList)
    
    return builder.generateSystem()

def haldaneHoneycombDisorder(W, H, t1Disorder, t2Disorder, Mdisorder, periodicBoundaryConditions=[], distribution = lambda x: np.random.normal(0,np.absolute(x))):
    """Returns the underlying lattice and built hamiltonian for a haldane 
    model on a honeycomb lattice of size W x H and nearest neighbor disorder t1, 
    next nearest neighbor disorder t2 (or t2.conj() as appropraite), and 
    sublattice potential M 
    """
    lattice = honeycombLattice(W,H,periodicBoundaryConditions)
    t2Disorderconj = np.conj(t2Disorder)
    
    disorderDisplacementList = []
    disorderMatrixList = []
    
    Dmatrix00 = np.zeros((2,2),dtype=complex)
    Dmatrix10 = np.zeros((2,2),dtype=complex)
    Dmatrix01 = np.zeros((2,2),dtype=complex)
    Dmatrix11 = np.zeros((2,2),dtype=complex)
    
    disorderDisplacementList.append([0,0])
    Dmatrix00[0][0] = Mdisorder
    Dmatrix00[0][1] = t1Disorder
    Dmatrix00[1][0] = t1Disorder
    Dmatrix00[1][1] = -Mdisorder
    disorderMatrixList.append(Dmatrix00)
    
    disorderDisplacementList.append([1,0])
    Dmatrix10[0][0] = t2Disorder
    Dmatrix10[0][1] = 0.0
    Dmatrix10[1][0] = t1Disorder
    Dmatrix10[1][1] = t2Disorderconj
    disorderMatrixList.append(Dmatrix10)
    
    disorderDisplacementList.append([0,1])
    Dmatrix01[0][0] = t2Disorderconj
    Dmatrix01[1][0] = t1Disorder
    Dmatrix01[0][1] = 0.0
    Dmatrix01[1][1] = t2Disorder
    disorderMatrixList.append(Dmatrix01)
    
    disorderDisplacementList.append([1,-1])
    Dmatrix11[0][0] = t2Disorderconj
    Dmatrix11[0][1] = 0.0
    Dmatrix11[1][0] = 0.0
    Dmatrix11[1][1] = t2Disorder
    disorderMatrixList.append(Dmatrix11)
    
    builder = TightBindingHamiltonianBuilder(lattice,disorderDisplacementList,disorderMatrixList,distribution)
    
    return builder.generateSystem()