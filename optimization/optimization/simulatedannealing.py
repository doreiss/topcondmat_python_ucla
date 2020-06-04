# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 15:38:55 2018

@author: dominic
"""

import numpy as np 
from random import random
import copy

class SimulatedAnnealing: 
    def __init__(self,temperatureFcn,
                 neighborFcn,energyFcn,probabilityFcn):
        """
        temperatureFcn is a function whcih takes the fraction of iterations
        as a parameter to simulate cooling without quenching
        
        neighborFcn is a function which takes the current state as a parameter
        and returns a candidate state to test
        
        energyFcn takes a state and evaluates its energy, i.e. the function
        we wish to optimize
        
        probabilityFcn takes the oldEnergy and the newEnergy and the current
        temperature and returns a probability between 0 and 1 of accepting the
        transition
        """
        self.temperature = temperatureFcn
        self.neighbor = neighborFcn
        self.energy = energyFcn
        self.probability = probabilityFcn
        
    def run(self,numIterations,initialState,verbose=False,saveEnergyHistory=False,
            saveStateHistory=False):
        """Runs the annealing from initialState for numIterations
        if verbose it prints the currentEnergy at each iteration
        Returns a pair (energy,state) where both energy and state may
        be either the final values or a history depending on function
        parameters
        """
        stateHistory = []
        energyHistory = []
        currentState = copy.deepcopy(initialState)
        currentEnergy = self.energy(currentState)
        for fraction in np.linspace(0,1,numIterations):
            if(saveStateHistory):
                stateHistory.append(currentState)
            if(saveEnergyHistory):
                energyHistory.append(currentEnergy)
            temp = self.temperature(fraction)
            newState = self.neighbor(currentState)
            newEnergy = self.energy(newState)
            if (self.probability(currentEnergy,newEnergy,temp) > random()):
                currentState = newState
                currentEnergy = newEnergy
                if(verbose):
                    print(currentEnergy)
        if(saveEnergyHistory and not saveStateHistory):    
            return energyHistory , currentState
        if(saveStateHistory and not saveEnergyHistory):
            return currentEnergy, stateHistory
        if(saveEnergyHistory and saveStateHistory):
            return energyHistory, stateHistory
        return currentEnergy, currentState

def naiveDescent(energyFcn,neighborFcn):
    
    def naiveTemp(anyState):
        return 0.0
    
    def naiveProbability(oldEnergy, newEnergy, temp):
        if(oldEnergy > newEnergy):
            return 1.0
        return 0.0
    
    return SimulatedAnnealing(naiveTemp,
                              neighborFcn, energyFcn, naiveProbability)