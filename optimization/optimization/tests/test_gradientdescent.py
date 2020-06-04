# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:11:50 2018

@author: dominic
"""
import numpy as np
from unittest import TestCase

import optimization.gradientdescent as gd

class TestGradientDescent(TestCase):
    def test_quadratic(self):
        def quadraticObjective(state):
            return state*state
        def quadraticGradient(state):
            return 2*state
        def quadraticStep(state,gradient,gamma):
            return state - gamma*gradient
        quadGD = gd.GradientDescent(quadraticObjective,quadraticGradient,quadraticStep)
        y, x, _ = quadGD.run(5.0)
        self.assertTrue(np.isclose(y,0.0,rtol=1e-05,atol=1e-05,equal_nan=False))
        self.assertTrue(np.isclose(x,0.0,rtol=1e-05,atol=1e-05,equal_nan=False))
        y, x, _ = quadGD.run(100.0)
        self.assertTrue(np.isclose(y,0.0,rtol=1e-05,atol=1e-05,equal_nan=False))
        self.assertTrue(np.isclose(x,0.0,rtol=1e-05,atol=1e-05,equal_nan=False))
        y, x, _ = quadGD.run(-25.0)
        self.assertTrue(np.isclose(y,0.0,rtol=1e-05,atol=1e-05,equal_nan=False))
        self.assertTrue(np.isclose(x,0.0,rtol=1e-05,atol=1e-05,equal_nan=False))
        y, x, _ = quadGD.run(1e06)
        self.assertTrue(np.isclose(y,0.0,rtol=1e-05,atol=1e-05,equal_nan=False))
        self.assertTrue(np.isclose(x,0.0,rtol=1e-05,atol=1e-05,equal_nan=False))
        y, x, _ = quadGD.run(0.0)
        self.assertTrue(np.isclose(y,0.0,rtol=1e-05,atol=1e-05,equal_nan=False))
        self.assertTrue(np.isclose(x,0.0,rtol=1e-05,atol=1e-05,equal_nan=False))
    def test_2Dquadratic(self):
        def quadraticObjective2D(state):
            return np.dot(state,state) + 2.0
        def quadraticGradient2D(state):
            return 2.0*state
        def quadraticStep2D(state,gradient,gamma):
            return state - gamma*gradient
        quadGD = gd.GradientDescent(quadraticObjective2D,quadraticGradient2D,quadraticStep2D)
        y, x, _ = quadGD.run(np.array([5.0,-100.0]))
        self.assertTrue(np.isclose(y,2.0,rtol=1e-05,atol=1e-05,equal_nan=False))
        self.assertTrue(np.all(np.isclose(x,np.zeros(2),rtol=1e-05,atol=1e-05,equal_nan=False)))
        y, x, _ = quadGD.run(np.array([100.0,75.0]))
        self.assertTrue(np.isclose(y,2.0,rtol=1e-05,atol=1e-05,equal_nan=False))
        self.assertTrue(np.all(np.isclose(x,np.zeros(2),rtol=1e-05,atol=1e-05,equal_nan=False)))
        y, x, _ = quadGD.run(np.array([5.0,-25.0]))
        self.assertTrue(np.isclose(y,2.0,rtol=1e-05,atol=1e-05,equal_nan=False))
        self.assertTrue(np.all(np.isclose(x,np.zeros(2),rtol=1e-05,atol=1e-05,equal_nan=False)))
        y, x, _ = quadGD.run(np.array([-5.0,-100.0]))
        self.assertTrue(np.isclose(y,2.0,rtol=1e-05,atol=1e-05,equal_nan=False))
        self.assertTrue(np.all(np.isclose(x,np.zeros(2),rtol=1e-05,atol=1e-05,equal_nan=False)))
        y, x, _ = quadGD.run(np.array([0.0,0.0]))
        self.assertTrue(np.isclose(y,2.0,rtol=1e-05,atol=1e-05,equal_nan=False))
        self.assertTrue(np.all(np.isclose(x,np.zeros(2),rtol=1e-05,atol=1e-05,equal_nan=False)))
        
    