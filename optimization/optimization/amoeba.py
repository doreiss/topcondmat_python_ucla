# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 21:37:51 2018

@author: dominic
"""
import bisect
import scipy
import numpy as np

class RiemannianAmoeba:
    def __init__(self,energyFcn,karchermeanFcn,geodesicFcn,expansionMaxFcn):
        """
        energyFcn is the function we wish to optimize which takes a state
        
        karchermeanFcn is a function which takes a list of states and an 
        initial guess for the mean and returns the state representing their 
        Karcher mean
        
        geodesicFcn takes two states p and q and returns a function parametrizing
        the geodesic between them s.t. geodesicFcn(p,q)(0) = p and geodesicFcn(p,q)
        (1) = q
        
        expansionMaxFcn is function whcih computes the maximum expansion factor given the
        orderedSimplex and mean
        """
        self.energy = energyFcn
        self.karchermean = karchermeanFcn
        self.geodesic = geodesicFcn
        self.orderedSimplex = []
        self.orderedEnergies = []
        self.expansionMax = expansionMaxFcn
        
    def run(self,numIterations,initialSimplex,verbose=False):
        energies = [self.energy(x) for x in initialSimplex]
        sortedEnergiesAndVertices = list(zip(energies,initialSimplex))
        sortedEnergiesAndVertices.sort(key = lambda t: t[0])
        self.orderedSimplex = [vertex for _,vertex in sortedEnergiesAndVertices]
        self.orderedEnergies = [energy for energy,_ in sortedEnergiesAndVertices]
    
        mean = self.orderedSimplex[0]
        
        for it in range(numIterations):
            
            reflectionRatio = -1.0
            expansionRatio = -2.0
            outsideContractionRatio = -0.5
            insideContractionRatio = 0.5
            shrinkingRatio = 0.5
            
            if(verbose):
                print(self.orderedEnergies[0])
                print(np.average(self.orderedEnergies))
                print(np.std(self.orderedEnergies))
                
            mean = self.karchermean(self.orderedSimplex[:-1],self.orderedSimplex[0])
            geodesicMeanLargest = self.geodesic(mean,self.orderedSimplex[-1])
            
            maxExpansion = self.expansionMax(self.orderedSimplex,mean)
            
            if(maxExpansion < 2.0):
                fraction = maxExpansion/2.0
                reflectionRatio *= fraction
                expansionRatio *= fraction
                outsideContractionRatio *= fraction
                insideContractionRatio *= fraction
                shrinkingRatio *= fraction          
            
            candidateReflection = geodesicMeanLargest(reflectionRatio)
            energyReflection = self.energy(candidateReflection)
            
            if(energyReflection > self.orderedEnergies[0] and
               energyReflection < self.orderedEnergies[-2]):
                
                self.orderedEnergies.pop()
                self.orderedSimplex.pop()
                insertionIndex = bisect.bisect_left(self.orderedEnergies,energyReflection)
                self.orderedEnergies.insert(insertionIndex,energyReflection)
                self.orderedSimplex.insert(insertionIndex,candidateReflection)
                if(verbose):
                    print('reflected')
                
            elif(energyReflection < self.orderedEnergies[0]):
                candidateExpansion = geodesicMeanLargest(expansionRatio)
                energyExpansion = self.energy(candidateExpansion)
                
                self.orderedEnergies.pop()
                self.orderedSimplex.pop()
                
                if(energyExpansion < energyReflection):
                    self.orderedEnergies.insert(0,energyExpansion)
                    self.orderedSimplex.insert(0,candidateExpansion)
                    if(verbose):
                        print('expanded')
                else:
                    self.orderedEnergies.insert(0,energyReflection)
                    self.orderedSimplex.insert(0,candidateReflection)
                    if(verbose):
                        print('reflected')
                    
            elif(energyReflection < self.orderedEnergies[-1] and 
                 energyReflection > self.orderedEnergies[-2]):
                candidateOutsideContraction = geodesicMeanLargest(outsideContractionRatio)
                energyOutsideContraction = self.energy(candidateOutsideContraction)
                if(energyOutsideContraction < energyReflection):
                    self.orderedEnergies.pop()
                    self.orderedSimplex.pop()
                    
                    insertionIndex = bisect.bisect_left(self.orderedEnergies,energyOutsideContraction)
                    self.orderedEnergies.insert(insertionIndex,energyOutsideContraction)
                    self.orderedSimplex.insert(insertionIndex,candidateOutsideContraction)
                    if(verbose):
                        print('outside contracted')
                else: 
                    self.__shrink(shrinkingRatio)
                    if(verbose):
                        print('shrunk')
            else:
                candidateInsideContraction = geodesicMeanLargest(insideContractionRatio)
                energyInsideContraction = self.energy(candidateInsideContraction)
                if(energyInsideContraction < self.orderedEnergies[-1]): 
                    self.orderedEnergies.pop()
                    self.orderedSimplex.pop()
                    
                    insertionIndex = bisect.bisect_left(self.orderedEnergies,energyInsideContraction)
                    self.orderedEnergies.insert(insertionIndex,energyInsideContraction)
                    self.orderedSimplex.insert(insertionIndex,candidateInsideContraction)
                    if(verbose):
                        print('inside contracted')
                else:
                    self.__shrink(shrinkingRatio)
                    if(verbose):
                        print('shrunk')
        return self.orderedEnergies[0], self.orderedSimplex[0]
                    
    def __shrink(self,ratio):
        geodesics = [self.geodesic(self.orderedSimplex[0],x) for x in self.orderedSimplex[1:]]
        newSimplex = [x(ratio) for x in geodesics]            
        newEnergies = [self.energy(vertex) for vertex in newSimplex]
        newSimplex.append(self.orderedSimplex[0])
        newEnergies.append(self.orderedEnergies[0])
        sortedEnergiesAndVertices = list(zip(newEnergies,newSimplex))
        sortedEnergiesAndVertices.sort(key = lambda t: t[0])
        self.orderedSimplex = [vertex for _,vertex in sortedEnergiesAndVertices]
        self.orderedEnergies = [energy for energy,_ in sortedEnergiesAndVertices]

def specialUnitaryAmoeba(energyFcn,n):
    """Creates an ameoba direct searcher on the special unitary group
    energyFcn is the fcn to minimize
    n is the square dimension of the SU matrices
    """
    def suKarcherMean(points,meanGuess):
        mean = meanGuess
        N = len(points)
        delta = 0.0001
        omega = 1.0/N*sum([scipy.linalg.logm(mean.conj().T@x) for x in points])
        omega = 0.5*(omega-omega.conj().T)
        while np.linalg.norm(omega) > delta :
            mean = mean@scipy.linalg.expm(omega)
            omega = 1.0/N*sum([scipy.linalg.logm(mean.conj().T@x) for x in points])
            omega = 0.5*(omega-omega.conj().T)
        mean = mean@scipy.linalg.expm(omega)
        return mean
    def suGeodesic(startPoint,endPoint):
        def geodesicFcn(t):
            log = scipy.linalg.logm(startPoint.conj().T@endPoint)
            log = 0.5*(log - log.conj().T)
            return startPoint@scipy.linalg.expm(-t*1j*log)
        return geodesicFcn
    def suExpansionMax(simplex,mean):
        logprod = scipy.linalg.logm(mean.conj().T@simplex[-1])
        logprod = 0.5*(logprod - logprod.conj().T)
        t = np.trace(logprod@logprod)
        return (np.pi/(4.0 * np.sqrt(-0.5*1.0/n*t))).real
    
    return RiemannianAmoeba(energyFcn,suKarcherMean,suGeodesic,suExpansionMax)
            
        
    