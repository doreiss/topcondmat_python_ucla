# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 11:09:09 2018

@author: dominic
"""

import numpy as np
import scipy
from unittest import TestCase

from mathtools.randommatrix import SmallUnitaryBuilder
from mathtools.randommatrix import randomHaarUnitary

class TestSmallUnitaryBuilder(TestCase):
    def test_isUnitary(self):
        n = 7
        t = .1
        builder = SmallUnitaryBuilder(n)
        u = builder.randomSmallUnitary(t)
        
        self.assertTrue(np.all(np.isclose(u@u.conj().T,np.identity(n,dtype=complex),rtol=1e-05, atol=1e-08, equal_nan=False)))
        self.assertTrue(np.all(np.isclose(u.conj().T@u,np.identity(n,dtype=complex),rtol=1e-05, atol=1e-08, equal_nan=False)))
        
    def test_isSmall(self):
        n = 7 
        t = .1
        builder = SmallUnitaryBuilder(n)
        u = builder.randomSmallUnitary(t)
        log = -1.0*1j*scipy.linalg.logm(u)
        
        self.assertTrue(np.all(np.isclose(log,0.5*(log + log.conj().T),rtol=1e-05, atol=1e-08, equal_nan=False)))
        self.assertTrue(np.isclose(builder.algebra.dot(log,log),t*t,rtol=1e-05, atol=1e-08, equal_nan=False))

class TestRandomHaarUnitary(TestCase):
    def test_isUnitary(self):
        n = 13
        u = randomHaarUnitary(n)
        self.assertTrue(np.all(np.isclose(u@u.conj().T,np.identity(n,dtype=complex),rtol=1e-05, atol=1e-08, equal_nan=False)))
        self.assertTrue(np.all(np.isclose(u.conj().T@u,np.identity(n,dtype=complex),rtol=1e-05, atol=1e-08, equal_nan=False)))