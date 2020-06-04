# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:51:55 2018

@author: dominic
"""

import numpy as np
from unittest import TestCase

from mathtools.liealgebra import SpecialUnitaryAlgebra

class TestSpecialUnitaryAlgebra(TestCase):
    def test_traceless(self):
        for N in range(10):
            algebra = SpecialUnitaryAlgebra(N)
            for basisElem in algebra.basis:
                self.assertTrue(np.isclose(np.trace(basisElem),0.0, atol=1e-08, equal_nan=False))
    
    def test_orthonormal(self):
        for N in range(10):
            algebra = SpecialUnitaryAlgebra(N)
            for index1, basisElem1 in enumerate(algebra.basis):
                for index2, basisElem2 in enumerate(algebra.basis):
                    if(index1 == index2):
                        self.assertTrue(np.isclose(algebra.dot(basisElem1,basisElem2),1.0,rtol=1e-05, atol=1e-08, equal_nan=False))
                    else:
                        self.assertTrue(np.isclose(algebra.dot(basisElem1,basisElem2),0.0,rtol=1e-05, atol=1e-08, equal_nan=False))