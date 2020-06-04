# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:36:40 2018

@author: dominic
"""
import numpy as np
from unittest import TestCase

import tightbinding.tightbindinghamiltonianbuilder as tbhb
import tightbinding.bravaislattice as bl

def pauliX():
	return np.array([[0,1],[1,0]],dtype=complex)
def pauliY():
	return np.array([[0,-1j],[1j,0]],dtype=complex)
def pauliZ():
	return  np.array([[1,0],[0,-1]])

class TestTightBindingHamiltonianBuilder(TestCase):
    def test_pbc_nearestneighbor_translationallyInvariant_real_1d(self):
        lattice = bl.oneDimensionalBipartiteLattice(4)
        hoppingDisplacementList = []
        hoppingMatrixList = []
		
        hoppingDisplacementList.append([0])
        hoppingMatrixList.append(pauliZ())
        
        hoppingDisplacementList.append([1])
		
        hoppingMatrixList.append(pauliX())
        
        builder = tbhb.TightBindingHamiltonianBuilder(lattice,hoppingDisplacementList,hoppingMatrixList)
        
        lat, ham = builder.generateSystem()
        
        row1 = np.array([1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0],dtype=complex)
        row2 = np.array([0.0,-1.0,1.0,0.0,0.0,0.0,1.0,0.0],dtype=complex)
        row3 = np.array([0.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0],dtype=complex)
        row4 = np.array([1.0,0.0,0.0,-1.0,1.0,0.0,0.0,0.0],dtype=complex)
        row5 = np.array([0.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0],dtype=complex)
        row6 = np.array([0.0,0.0,1.0,0.0,0.0,-1.0,1.0,0.0],dtype=complex)
        row7 = np.array([0.0,1.0,0.0,0.0,0.0,1.0,1.0,0.0],dtype=complex)
        row8 = np.array([1.0,0.0,0.0,0.0,1.0,0.0,0.0,-1.0],dtype=complex)
        
        e_ham = np.array([row1,row2,row3,row4,row5,row6,row7,row8],dtype=complex)

        self.assertTrue(np.all(np.isclose(ham,e_ham,rtol=1e-05, atol=1e-08, equal_nan=False)))
        
        
    def test_npbc_nearestneighbor_translationallyInvariant_real_1d(self):
        lattice = bl.oneDimensionalBipartiteLattice(4,[False])
        hoppingDisplacementList = []
        hoppingMatrixList = []
        
        hoppingDisplacementList.append([0])
        hoppingMatrixList.append(pauliZ())
        
        hoppingDisplacementList.append([1])
        hoppingMatrixList.append(pauliX())
        
        builder = tbhb.TightBindingHamiltonianBuilder(lattice,hoppingDisplacementList,hoppingMatrixList)
        
        lat, ham = builder.generateSystem()
        
        row1 = np.array([1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0],dtype=complex)
        row2 = np.array([0.0,-1.0,1.0,0.0,0.0,0.0,0.0,0.0],dtype=complex)
        row3 = np.array([0.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0],dtype=complex)
        row4 = np.array([1.0,0.0,0.0,-1.0,1.0,0.0,0.0,0.0],dtype=complex)
        row5 = np.array([0.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0],dtype=complex)
        row6 = np.array([0.0,0.0,1.0,0.0,0.0,-1.0,1.0,0.0],dtype=complex)
        row7 = np.array([0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0],dtype=complex)
        row8 = np.array([0.0,0.0,0.0,0.0,1.0,0.0,0.0,-1.0],dtype=complex)
        
        e_ham = np.array([row1,row2,row3,row4,row5,row6,row7,row8],dtype=complex)

        self.assertTrue(np.all(np.isclose(ham,e_ham,rtol=1e-05, atol=1e-08, equal_nan=False)))
    
    def test_pbc_nearestneighbor_translationallyInvariant_complex_1d(self):
        lattice = bl.oneDimensionalBipartiteLattice(4)
        hoppingDisplacementList = []
        hoppingMatrixList = []
        
        hoppingDisplacementList.append([0])
        hoppingMatrixList.append(pauliZ())
        
        hoppingDisplacementList.append([1])
        hoppingMatrixList.append(pauliY())
        
        builder = tbhb.TightBindingHamiltonianBuilder(lattice,hoppingDisplacementList,hoppingMatrixList)
        
        lat, ham = builder.generateSystem()
        
        row1 = np.array([1,0,0,-1j,0,0,0,-1j],dtype=complex)
        row2 = np.array([0,-1,1j,0,0,0,1j,0],dtype=complex)
        row3 = np.array([0,-1j,1,0,0,-1j,0,0],dtype=complex)
        row4 = np.array([1j,0,0,-1,1j,0,0,0],dtype=complex)
        row5 = np.array([0,0,0,-1j,1,0,0,-1j],dtype=complex)
        row6 = np.array([0,0,1j,0,0,-1,1j,0],dtype=complex)
        row7 = np.array([0,-1j,0,0,0,-1j,1,0],dtype=complex)
        row8 = np.array([1j,0,0,0,1j,0,0,-1],dtype=complex)
        
        e_ham = np.array([row1,row2,row3,row4,row5,row6,row7,row8],dtype=complex)
        self.assertTrue(np.all(np.isclose(ham,e_ham,rtol=1e-05, atol=1e-08, equal_nan=False)))
    
    def test_npbc_nearestneighbor_translationallyInvariant_complex(self):
        lattice = bl.oneDimensionalBipartiteLattice(4,[False])
        hoppingDisplacementList = []
        hoppingMatrixList = []
        
        hoppingDisplacementList.append([0])
        hoppingMatrixList.append(pauliZ())
        
        hoppingDisplacementList.append([1])
        hoppingMatrixList.append(pauliY())
        
        builder = tbhb.TightBindingHamiltonianBuilder(lattice,hoppingDisplacementList,hoppingMatrixList)
        
        lat, ham = builder.generateSystem()
        
        row1 = np.array([1,0,0,-1j,0,0,0,0],dtype=complex)
        row2 = np.array([0,-1,1j,0,0,0,0,0],dtype=complex)
        row3 = np.array([0,-1j,1,0,0,-1j,0,0],dtype=complex)
        row4 = np.array([1j,0,0,-1,1j,0,0,0],dtype=complex)
        row5 = np.array([0,0,0,-1j,1,0,0,-1j],dtype=complex)
        row6 = np.array([0,0,1j,0,0,-1,1j,0],dtype=complex)
        row7 = np.array([0,0,0,0,0,-1j,1,0],dtype=complex)
        row8 = np.array([0,0,0,0,1j,0,0,-1],dtype=complex)
        
        e_ham = np.array([row1,row2,row3,row4,row5,row6,row7,row8],dtype=complex)
        self.assertTrue(np.all(np.isclose(ham,e_ham,rtol=1e-05, atol=1e-08, equal_nan=False)))
     
    def test_pbc_nnearestneighbor_translationallyInvariant_1d(self):
        lattice = bl.oneDimensionalBipartiteLattice(5)
        hoppingDisplacementList = []
        hoppingMatrixList = []
        
        hoppingDisplacementList.append([0])
        hoppingMatrixList.append(pauliZ())
        
        hoppingDisplacementList.append([2])
        hoppingMatrixList.append(pauliY())
        
        builder = tbhb.TightBindingHamiltonianBuilder(lattice,hoppingDisplacementList,hoppingMatrixList)
        
        lat, ham = builder.generateSystem()
        
        row1  = np.array([1,0,0,0,0,-1j,0,-1j,0,0],dtype=complex)
        row2  = np.array([0,-1,0,0,1j,0,1j,0,0,0],dtype=complex)
        row3  = np.array([0,0,1,0,0,0,0,-1j,0,-1j],dtype=complex)
        row4  = np.array([0,0,0,-1,0,0,1j,0,1j,0],dtype=complex)
        row5  = np.array([0,-1j,0,0,1,0,0,0,0,-1j],dtype=complex)
        row6  = np.array([1j,0,0,0,0,-1,0,0,1j,0],dtype=complex)
        row7  = np.array([0,-1j,0,-1j,0,0,1,0,0,0],dtype=complex)
        row8  = np.array([1j,0,1j,0,0,0,0,-1,0,0],dtype=complex)
        row9  = np.array([0,0,0,-1j,0,-1j,0,0,1,0],dtype=complex)
        row10 = np.array([0,0,1j,0,1j,0,0,0,0,-1],dtype=complex)

        e_ham = np.array([row1,row2,row3,row4,row5,row6,row7,row8,row9,row10],dtype=complex)
        self.assertTrue(np.all(np.isclose(ham,e_ham,rtol=1e-05, atol=1e-08, equal_nan=False)))
    
    def test_npbc_nnearestneighbor_translationallyInvariant_1d(self):
        lattice = bl.oneDimensionalBipartiteLattice(5,[False])
        hoppingDisplacementList = []
        hoppingMatrixList = []
        
        hoppingDisplacementList.append([0])
        hoppingMatrixList.append(pauliZ())
        
        hoppingDisplacementList.append([2])
        hoppingMatrixList.append(pauliY())
        
        builder = tbhb.TightBindingHamiltonianBuilder(lattice,hoppingDisplacementList,hoppingMatrixList)
        
        lat, ham = builder.generateSystem()
        
        row1 = np.array([1,0,0,0,0,-1j,0,0,0,0],dtype=complex)
        row2 = np.array([0,-1,0,0,1j,0,0,0,0,0],dtype=complex)
        row3 = np.array([0,0,1,0,0,0,0,-1j,0,0],dtype=complex)
        row4 = np.array([0,0,0,-1,0,0,1j,0,0,0],dtype=complex)
        row5 = np.array([0,-1j,0,0,1,0,0,0,0,-1j],dtype=complex)
        row6 = np.array([1j,0,0,0,0,-1,0,0,1j,0],dtype=complex)
        row7 = np.array([0,0,0,-1j,0,0,1,0,0,0],dtype=complex)
        row8 = np.array([0,0,1j,0,0,0,0,-1,0,0],dtype=complex)
        row9 = np.array([0,0,0,0,0,-1j,0,0,1,0],dtype=complex)
        row10 = np.array([0,0,0,0,1j,0,0,0,0,-1],dtype=complex)
        
        e_ham = np.array([row1,row2,row3,row4,row5,row6,row7,row8,row9,row10],dtype=complex)
        self.assertTrue(np.all(np.isclose(ham,e_ham,rtol=1e-05, atol=1e-08, equal_nan=False)))