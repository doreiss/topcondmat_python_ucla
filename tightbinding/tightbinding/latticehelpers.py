# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:24:48 2018

@author: roygroup-nuc01
"""
import numpy as np
from mathtools.randommatrix import randomHaarUnitary

class LatticeUnitaryPlacer:
    """Places local unitary matrices in the correct sector 
    of larger matrices within lattices
    """
    def __init__(self,bravaisLattice):
        self.bravaisLattice = bravaisLattice
    
    def placeUnitary(self,unitary,siteIndexList):
        numSites = len(siteIndexList)
        N = self.bravaisLattice.onsiteDimension
        U = np.identity(self.bravaisLattice.hilbertSpaceDimension,complex)
        
        initialStateIndexList = []
        finalStateIndexList = []
        
        for siteIndex in siteIndexList:
            initialStateIndexList.append(self.bravaisLattice.stateIndex(siteIndex))
            finalStateIndexList.append(self.bravaisLattice.stateIndex(siteIndex,N-1))
            
        for i in range(numSites):
            for j in range(i+1):
                U[initialStateIndexList[i]:finalStateIndexList[i]+1,
                  initialStateIndexList[j]:finalStateIndexList[j]+1] = unitary[i*N:(i+1)*N,j*N:(j+1)*N]
                if( i != j):
                    U[initialStateIndexList[j]:finalStateIndexList[j]+1,
                      initialStateIndexList[i]:finalStateIndexList[i]+1] = unitary[j*N:(j+1)*N,i*N:(i+1)*N]
        
        return U
		
class LocalHaarUnitaryBuilder:
    """Builds random unitary operators associated with an underlying lattice
    """
    def __init__(self,bravaisLattice):
        self.bravaisLattice = bravaisLattice
        
    def twoSiteUnitary(self,siteIndexA, siteIndexB):
        """Generates a random unitary between two sites and the identity
        elsewhere
        """
        N = self.bravaisLattice.onsiteDimension
        U = np.identity(self.bravaisLattice.hilbertSpaceDimension,complex)
        randomU = randomHaarUnitary(2*N)

        initialStateIndexA = self.bravaisLattice.stateIndex(siteIndexA)
        initialStateIndexB = self.bravaisLattice.stateIndex(siteIndexB)
        finalStateIndexA = self.bravaisLattice.stateIndex(siteIndexA,N-1)
        finalStateIndexB = self.bravaisLattice.stateIndex(siteIndexB,N-1)
        
        U[initialStateIndexA:finalStateIndexA+1, 
          initialStateIndexA:finalStateIndexA+1] = randomU[0:N,0:N]
        U[initialStateIndexA:finalStateIndexA+1,
          initialStateIndexB:finalStateIndexB+1] = randomU[0:N,N:2*N]
        U[initialStateIndexB:finalStateIndexB+1,
          initialStateIndexA:finalStateIndexA+1] = randomU[N:2*N,0:N]
        U[initialStateIndexB:finalStateIndexB+1,
          initialStateIndexB:finalStateIndexB+1] = randomU[N:2*N,N:2*N]
        
        return U
    
    def nSiteUnitary(self,siteIndexList):
        """Generates a random unitary between the sites contained in siteIndexList
        and the identity elsewhere
        """
        numSites = len(siteIndexList)
        N = self.bravaisLattice.onsiteDimension
        U = np.identity(self.bravaisLattice.hilbertSpaceDimension,complex)
        randomU = randomHaarUnitary(numSites*N)
        
        initialStateIndexList = []
        finalStateIndexList = []
        
        for siteIndex in siteIndexList:
            initialStateIndexList.append(self.bravaisLattice.stateIndex(siteIndex))
            finalStateIndexList.append(self.bravaisLattice.stateIndex(siteIndex,N-1))
            
        for i in range(numSites):
            for j in range(i+1):
                U[initialStateIndexList[i]:finalStateIndexList[i]+1,
                  initialStateIndexList[j]:finalStateIndexList[j]+1] = randomU[i*N:(i+1)*N,j*N:(j+1)*N]
                if( i != j):
                    U[initialStateIndexList[j]:finalStateIndexList[j]+1,
                      initialStateIndexList[i]:finalStateIndexList[i]+1] = randomU[j*N:(j+1)*N,i*N:(i+1)*N]
        
        return U
            
    def randomTwoNeighborUnitary(self):
        """Chooses two random nearest neighbors in the lattice and chooses
        a random unitary between them and the identity elsewhere
        """
        randomSite = randrange(0,self.bravaisLattice.numSites)
        randomNeighbor = -1
        while(randomNeighbor == -1):
            randomDirection = randrange(0,self.bravaisLattice.spatialDimension)
            randomSign = 1 if random() < 0.5 else -1
            randomNeighbor = self.bravaisLattice.getNeighborSiteIndex(randomSite,randomDirection,randomSign)
        return self.twoSiteUnitary(randomSite,randomNeighbor)
    
    def randomNeighborhoodUnitary(self,neighborhoodMaxSize):
        """Finds a random site and a random neighborhood of size less than
        neighborhoodMaxSize and finds a unitary in that neighborhood and the 
        identity elsewhere
        """
        randomSite = randrange(0,self.bravaisLattice.numSites)
        randSize = randrange(0,neighborhoodMaxSize+1)
        return self.nSiteUnitary(self.bravaisLattice.sitesWithinNSteps(randomSite,randSize))