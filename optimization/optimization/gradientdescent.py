# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:42:45 2018

@author: dominic
"""

import copy
import numpy as np

class GradientDescent:
    def __init__(self, objectiveFcn, gradientFcn, stepFcn):
        """
        gradientFcn is a function which takes a point in the state space and returns a 
        vector representing the gradient according to some basis
        
        stepFcn takes the current state, gradient, and gamma as argument and produces the 
        next state for iteration
        """
        self.objective = objectiveFcn
        self.gradient = gradientFcn
        self.createStep = stepFcn
    
    def run(self, initialState, gradPrecision= 1e-05, minimumStepDifference = 1e-05,
            initialGamma = 0.1, maxIterations = 1e06, verbose = False, minGamma = 1e-08):
        
        currentState = copy.deepcopy(initialState)
        currentObj = self.objective(currentState)
        currentGradient = self.gradient(currentState)
        currentGradNorm = np.linalg.norm(currentGradient)
        
        previousState = copy.deepcopy(currentState)
        previousObj = copy.deepcopy(currentObj)
        previousGradNorm = copy.deepcopy(currentGradNorm)
        
        gamma = copy.deepcopy(initialGamma)
        
        gradientNorm = np.linalg.norm(currentGradient)
        
        iteration = 0
        currentDiff = 0
        
        currentObj = self.objective(currentState)
        
        previousSuccessfulDiff = 0
        
        while(gradientNorm > gradPrecision and iteration < maxIterations and gamma > minGamma):
            
            currentGradient = self.gradient(currentState)
            currentGradNorm = np.linalg.norm(currentGradient)
            currentState = self.createStep(currentState,currentGradient,gamma)
            
            
            currentObj = self.objective(currentState)
            currentDiff = previousObj - currentObj
            
            if(currentDiff < 0):
                gamma *= 0.95
                currentState = copy.deepcopy(previousState)
                
                if(verbose):
                    print("Step too large: now gamma is ", gamma)
            elif(currentDiff > previousSuccessfulDiff or currentDiff < minimumStepDifference or currentGradNorm > previousGradNorm):
                previousSuccessfulDiff = currentDiff
                gamma *= 1.05
                previousState = copy.deepcopy(currentState)
                previousObj = copy.deepcopy(currentObj)
                previousGradNorm = copy.deepcopy(currentGradNorm)
                
                if(verbose): 
                    print("Step too small: now gamma is ", gamma)
            else:
                previousSuccessfulDiff = currentDiff
                previousState = copy.deepcopy(currentState)
                previousObj = copy.deepcopy(currentObj)
                previousGradNorm = copy.deepcopy(currentGradNorm)
            
            gradientNorm = np.linalg.norm(currentGradient)
            iteration = iteration + 1
            if(verbose):     
                print("At iteration: ", iteration, " gradient norm is: ", gradientNorm)
            if(iteration % 250 == 0): 
                print("On iteration #",iteration,": obj = ", currentObj, ", gradNorm = ", gradientNorm)
        if(verbose):
            print("Ended at ", iteration, " steps with a graident norm of ", gradientNorm, " and obj function value of ", currentObj, ".")
        return currentObj, currentState, iteration