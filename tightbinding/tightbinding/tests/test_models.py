# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:35:41 2018

@author: dominic
"""
import numpy as np
from unittest import TestCase
from tightbinding.models import kitaevonedchain
from tightbinding.models import haldane
class TestTightBindingModels(TestCase):
    def test_kiatev1dZeroModes(self):
        L = 20
        mu = 0
        t = 1.0
        delta = 1.0
        latpbc, hampbc = kitaevonedchain.Kitaev1dChain(L,mu,t,delta)
        latnpbc, hamnpbc = kitaevonedchain.Kitaev1dChain(L,mu,t,delta,[False])
        
        wpbc = np.linalg.eigvalsh(hampbc)
        wnpbc = np.linalg.eigvalsh(hamnpbc)
        
        self.assertTrue(not np.any(np.isclose(wpbc,np.zeros((2*L),dtype=complex),rtol=1e-05, atol=1e-08, equal_nan=False)))
        self.assertTrue(np.any(np.isclose(wnpbc,np.zeros((2*L),dtype=complex),rtol=1e-05, atol=1e-08, equal_nan=False)))
    def test_kitaev1dSpectrum(self):
        L = 21
        mu = 0.5
        t = 1.1
        delta = 0.9
        latpbc, hampbc = kitaevonedchain.Kitaev1dChain(L,mu,t,delta)
        spectrum = np.linalg.eigvalsh(hampbc)
        expectedSpectrum = np.load('tightbinding/tests/kitaev1dTestSpectrum0.npy') 
        print(spectrum)
        print(expectedSpectrum)
        self.assertTrue(np.all(np.isclose(spectrum,expectedSpectrum,rtol=1e-05, atol=1e-08, equal_nan=False)))
        
    def test_haldaneHoneycombSpectrum(self):
        W = 30
        H = 30
        t1 = 1.0
        t2 = 0.03*1j
        m = 0.0
        lat, ham = haldane.haldaneHoneycombSystem(W,H,t1,t2,m,[True,True])
        
        spectrum = np.linalg.eigvalsh(ham)
        expectedSpectrum = np.load('tightbinding/tests/haldaneHoneycombTestSpectrum0.npy')      
        self.assertTrue(np.all(np.isclose(spectrum,expectedSpectrum,rtol=1e-05, atol=1e-08, equal_nan=False)))
        
        W = 30
        H = 30
        t1 = 1.0
        t2 = 0.03*(np.cos(np.pi/3.0) + np.sin(np.pi/3.0)*1j)
        m = 0.0
        lat, ham = haldane.haldaneHoneycombSystem(W,H,t1,t2,m,[True,True])
        
        spectrum = np.linalg.eigvalsh(ham)
        expectedSpectrum = np.load('tightbinding/tests/haldaneHoneycombTestSpectrum1.npy')      
        self.assertTrue(np.all(np.isclose(spectrum,expectedSpectrum,rtol=1e-05, atol=1e-08, equal_nan=False)))