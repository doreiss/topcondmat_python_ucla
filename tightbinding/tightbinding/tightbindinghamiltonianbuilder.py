# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 11:29:14 2018

@author: dominic and alberto
"""

import numpy as np

class TightBindingHamiltonianBuilder:
    def __init__(self, bravaisLattice, hoppingDisplacementList, hoppingMatrixList, hoppingRandomDistributionFunction = (lambda x: x)):
        """Builds a tight binding hamiltonian. 
        hoppingDisplacementList is a list of displacement vectors which appear in 
        the translationally invariant part of the hamiltonian. These correspond to 
        hopping elements which exist. For example if it includes [1,0], this means
        there is a hopping term in the positive x direction. One should not include
        the redundant terms, for example if you include [1,0], do not include [-1,0]
        as the values of these amplitudes are implied by hermiticity. 
        
        hoppingMatrixlist is a list of hopping matrices, square and of dimension
        the onsite hilbert space dimension. They are ordered with respect to the
        previous list of displacements. 
        """
        self.bravaisLattice = bravaisLattice
        self.hoppingDisplacementList = hoppingDisplacementList
        self.hoppingMatrixList = hoppingMatrixList
        self.hoppingRandomDistributionFunction = np.vectorize(hoppingRandomDistributionFunction)
   
    def generateHamiltonain(self):
        """Generates different pieces of Hamiltonian and adds them, for example
        translationally invariant pieces, disorder pieces etc...
        """
        dim = self.bravaisLattice.hilbertSpaceDimension
        ham = np.zeros((dim,dim),complex)
        ham += self.__generateTranslationallyInvariantHamiltonian()
        #We expect other pieces of code to go here to generate disordered or 
        #auxillary parts of the hamiltonian in the future
        return ham
    
    def generateSystem(self):
        """Returns the Hamiltonian and associated lattice
        """
        return self.bravaisLattice, self.generateHamiltonain()
    
    def __generateTranslationallyInvariantHamiltonian(self):
        """Private helper function which generates the translationally invariant
        part of a hamiltonian for the builder. 
        """
        dim = self.bravaisLattice.hilbertSpaceDimension
        ham = np.zeros((dim,dim),dtype=complex)
        for siteIndex in range(self.bravaisLattice.numSites):
            for displacementIndex, displacementVector in enumerate(self.hoppingDisplacementList):
                otherSiteIndex = self.bravaisLattice.getRelativeSiteIndex(siteIndex,displacementVector)
                if(not otherSiteIndex == -1):
                    thisStateIndex = self.bravaisLattice.stateIndex(siteIndex)
                    otherStateIndex = self.bravaisLattice.stateIndex(otherSiteIndex)
                    
                    hoppingDisorderInst = self.hoppingRandomDistributionFunction(self.hoppingMatrixList[displacementIndex])
                    hoppingDisorderInstConj = hoppingDisorderInst.conj().T
                
                    ham[otherStateIndex:otherStateIndex+self.bravaisLattice.onsiteDimension,thisStateIndex:thisStateIndex+self.bravaisLattice.onsiteDimension] = hoppingDisorderInst
                
                    if(thisStateIndex != otherStateIndex):
                        ham[thisStateIndex:thisStateIndex+self.bravaisLattice.onsiteDimension,otherStateIndex:otherStateIndex+self.bravaisLattice.onsiteDimension] = hoppingDisorderInstConj
        return ham

