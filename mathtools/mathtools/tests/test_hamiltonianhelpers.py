# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:11:50 2018

@author: dominic
"""
import numpy as np
from unittest import TestCase

import mathtools.hamiltonianhelpers as hh

class TestHamiltonianHelpers(TestCase):
    def test_pauliMatrices(self):
        pX = np.array([[0.0,1.0],[1.0,0.0]],dtype=complex)
        pY = np.array([[0.0,-1.0*1j],[1.0*1j,0.0]],dtype=complex)
        pZ = np.array([[1.0,0.0],[0.0,-1.0]],dtype=complex)
        
        self.assertTrue(np.all(pX==hh.pauliX()))
        self.assertTrue(np.all(pY==hh.pauliY()))
        self.assertTrue(np.all(pZ==hh.pauliZ()))
    
    def test_projector(self):
        real_vec = np.array([1.0/np.sqrt(2.0),-1.0/np.sqrt(2.0),0.0],dtype=complex)
        real_proj = np.array([[0.5,-0.5,0.0],[-0.5,0.5,0.0],[0.0,0.0,0.0]],dtype=complex)
        self.assertTrue(np.all(np.isclose(real_proj, hh.projector(real_vec), rtol=1e-05, atol=1e-08, equal_nan=False)))
        
        complex_vec = np.array([1.0/np.sqrt(3.0)*1j,-1.0/np.sqrt(3.0),-1.0/np.sqrt(3.0)*1j],dtype=complex)
        complex_proj = np.array([[1.0/3.0,-1.0/3.0*1j,-1.0/3.0],[1.0/3.0*1j,1.0/3.0,-1.0/3.0*1j],[-1.0/3.0,1.0/3.0*1j,1.0/3.0]],dtype=complex)        
        self.assertTrue(np.all(np.isclose(complex_proj, hh.projector(complex_vec), rtol=1e-05, atol=1e-08, equal_nan=False)))
        
    def test_commutator(self):
        pX = hh.pauliX()
        pY = hh.pauliY()
        pZ = hh.pauliZ()
        
        self.assertTrue(np.all(np.isclose(hh.commutator(pX,pY),2.0*1j*pZ,rtol=1e-05, atol=1e-08, equal_nan=False)))
        
    #should add rest of hh tests here