# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:01:52 2018

@author: dominic
"""
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class BravaisLattice:
    def __init__(self, basisVectors, dimensionLengths, unitcellBasis = np.array([(0,0)]),periodicBoundaryConditions=[]): 
        """Creates a Bravais Lattice with basis vectors
        basisVectors: square numpy array with elements the BL basis vectors
        dimensionLengths: list of number of sites per spatial dimension
        numSites: total number of sites
        sitePositions: list of numpy arrays with site positions
        """
        self.basisVectors = basisVectors.A
        self.basisVectorNorms = [np.linalg.norm(bv) for bv in self.basisVectors]
        self.inverseTransform = np.linalg.inv(self.basisVectors)
        self.spatialDimension = len(basisVectors)
        
        if(len(periodicBoundaryConditions)==0):
            periodicBoundaryConditions = [True]*self.spatialDimension
        self.periodicBoundaryConditions = periodicBoundaryConditions
        
        self.dimensionLengths = dimensionLengths
        self.numSites = np.prod(dimensionLengths)
        self.dimensionPhysicalLengths = [a*b for a,b in zip(self.dimensionLengths,self.basisVectorNorms)]
        
        self.sitePositions = []
        self.nearestNeighbors = []
        for siteIndex in range(self.numSites):
            siteIntegerVec = self.siteIntegerVector(siteIndex)
            self.sitePositions.append(np.multiply(self.basisVectors, siteIntegerVec[:,np.newaxis]).sum(axis=0))
            self.nearestNeighbors.append(self._findNearestNeighbors(siteIndex))
        
        self.onsiteDimension = len(unitcellBasis)
        self.unitcellBasis = unitcellBasis
        self.hilbertSpaceDimension = self.numSites * self.onsiteDimension
        
        self.statePositions = []
        for stateIndex in range(self.hilbertSpaceDimension):
            siteIndex, orbitalIndex = self.siteAndOrbitalIndex(stateIndex)
            self.statePositions.append(self.sitePositions[siteIndex]+self.unitcellBasis[orbitalIndex])
        
    def siteIndex(self,siteVector): 
        """Returns an index between 0 and numSites-1 corresponding to
        siteVector. For example in a 2x2x2 system, the siteVector [1,1,0] 
        will return 3
        """
        siteInd = 0
        nPowSteps = 1
        for n in range(self.spatialDimension): 
            siteInd += (siteVector[n]*nPowSteps)
            nPowSteps *= self.dimensionLengths[n]
        return siteInd
    
    def siteIntegerVector(self,siteInd):
        """Returns the integer vector of a site with index siteInd
        """
        siteIntVec = []
        for dimension in range(self.spatialDimension):
            siteIntVec.append(self._siteDimStep(dimension,siteInd))
        return np.array(siteIntVec,dtype=int)
    
    def stateIndex(self,siteInd,orbitalInd=0):
        """Returns the state index (from 0 - hilbertspacedim - 1)
        given the site index and the orbital index
        """
        return (siteInd*self.onsiteDimension + orbitalInd) 
        
    def siteAndOrbitalIndex(self,stateIndex):
        """Returns the site index, orbital index given a state index
        """
        siteIndex =  stateIndex // self.onsiteDimension
        orbitalIndex = stateIndex % self.onsiteDimension
        return siteIndex, orbitalIndex
    
    def getNeighborSiteIndex(self,siteIndex,dimensionIndex,plusOrMinus):
        """Returns the site index of nearest neighbor of siteIndex 
        in the plusOrMinus*dimensionIndex direction
        """
        if(siteIndex == -1):
            return -1
        neighborIndex = 2*dimensionIndex
        if(plusOrMinus == -1): 
            neighborIndex += 1
        return self.nearestNeighbors[siteIndex][neighborIndex]
    
    def getRelativeSiteIndex(self,siteIndex,relativeSiteVector):
        """Returns the site index of site displaced from siteIndex
        by the relativeSiteVector
        """
        newSiteIndex = siteIndex
        for dimension in range(self.spatialDimension):
            if (newSiteIndex == -1):
                break
            sign = int(np.sign(relativeSiteVector[dimension]))
            for step in range(int(abs(relativeSiteVector[dimension]))):
                newSiteIndex = self.getNeighborSiteIndex(newSiteIndex,dimension,sign)
                if (newSiteIndex == -1): 
                    break
        return newSiteIndex
    
    def shortestSiteDisplacementVector(self,siteIndexFrom,siteIndexTo):
        """Returns the shortest relative site displacement vector
        from siteIndexFrom to siteIndexTo. If the system has open boundary
        conditions in all dimensions this is always the same as the
        relative displacement vector
        """
        
        siteIntVecFrom = self.siteIntegerVector(siteIndexFrom)
        siteIntVecTo   = self.siteIntegerVector(siteIndexTo)
        
        shortestSiteDisplacementVector = []
        
        for dimension in range(self.spatialDimension):
            normalDisplacement = siteIntVecTo[dimension] - siteIntVecFrom[dimension]
            wrappedDisplacement = -np.sign(normalDisplacement)*np.abs(np.abs(normalDisplacement) - self.dimensionLengths[dimension])
            
            if(not self.periodicBoundaryConditions[dimension]):
                shortestSiteDisplacementVector.append(normalDisplacement)
            else:
                if(np.abs(normalDisplacement) < np.abs(wrappedDisplacement)):
                    shortestSiteDisplacementVector.append(normalDisplacement)    
                else:
                    shortestSiteDisplacementVector.append(wrappedDisplacement)
        
        return np.array(shortestSiteDisplacementVector, dtype=int)
    
    def distanceSquaredMatrix(self,fromSite):
        """Returns a diagonal matrix whose elements are the sitewise distance
        squared from fromSite
        """
        distSquaredMat = np.zeros((self.hilbertSpaceDimension,self.hilbertSpaceDimension),dtype=complex)
        for siteIndex in range(self.numSites):
            dist = np.linalg.norm(self.shortestSiteDisplacementVector(fromSite,siteIndex))
            diagonalSubmatrix = np.eye(self.onsiteDimension,dtype=complex)
            diagonalSubmatrix = (dist*dist)*diagonalSubmatrix
            fromIndex = self.onsiteDimension*siteIndex
            toIndex = fromIndex + self.onsiteDimension
            distSquaredMat[fromIndex:toIndex,fromIndex:toIndex] = diagonalSubmatrix
        return distSquaredMat
    
    def balancedDistanceSquaredFromOrigin(self):
        """Returns a diagonal matrix whose elements are the sitewise distance
        squared from fromSite
        """
        dispMats = [np.dot(self.balancedDisplacementInDimFromOrigin(dim),self.balancedDisplacementInDimFromOrigin(dim)) for dim in range(self.spatialDimension)]
        return sum(dispMats)
    
    def displacementInDim(self,fromSite,dimension):
        """Returns a diagonal matrix whose elements are the shortest displacement to
        fromSite along dimension"""
        dispMat = np.zeros((self.hilbertSpaceDimension,self.hilbertSpaceDimension),dtype=complex)
        for siteIndex in range(self.numSites):
            disp = self.shortestSiteDisplacementVector(fromSite,siteIndex)[dimension]
            diagonalSubmatrix = np.eye(self.onsiteDimension,dtype=complex)
            diagonalSubmatrix = disp*diagonalSubmatrix
            fromIndex = self.onsiteDimension*siteIndex
            toIndex = fromIndex + self.onsiteDimension
            dispMat[fromIndex:toIndex,fromIndex:toIndex] = diagonalSubmatrix
        return dispMat
    
    def balancedDisplacementInDimFromOrigin(self,dimension):
        """Returns a diagonal matrix whose elements are the balanced displacement to
        origin along dimension"""
        dispMat = np.zeros((self.hilbertSpaceDimension,self.hilbertSpaceDimension),dtype=complex)
        for siteIndex in range(self.numSites):
            disp = self.shortestSiteDisplacementVector(0,siteIndex)[dimension]
            if(self.dimensionLengths[dimension] % 2 == 0):
                disp = disp + 0.5
                if disp > self.dimensionLengths[dimension] // 2: 
                    disp -= self.dimensionLengths[dimension]
            diagonalSubmatrix = np.eye(self.onsiteDimension,dtype=complex)
            diagonalSubmatrix = disp*diagonalSubmatrix
            fromIndex = self.onsiteDimension*siteIndex
            toIndex = fromIndex + self.onsiteDimension
            dispMat[fromIndex:toIndex,fromIndex:toIndex] = diagonalSubmatrix
        return dispMat
    
    def displacementSquaredInDim(self,fromSite,dimension):
        """Returns a diagonal matrix whose elements are the displacement to
        fromSite along dimension"""
        dispMat = np.zeros((self.hilbertSpaceDimension,self.hilbertSpaceDimension),dtype=complex)
        for siteIndex in range(self.numSites):
            disp = self.shortestSiteDisplacementVector(fromSite,siteIndex)[dimension]
            diagonalSubmatrix = np.eye(self.onsiteDimension,dtype=complex)
            diagonalSubmatrix = disp*disp*diagonalSubmatrix
            fromIndex = self.onsiteDimension*siteIndex
            toIndex = fromIndex + self.onsiteDimension
            dispMat[fromIndex:toIndex,fromIndex:toIndex] = diagonalSubmatrix
        return dispMat
    

    def sitesWithinNSteps(self,siteOrigin,N):
        """Returns a list of site indexes corresponding to sites within N
        nearest neighbor steps of the site at index siteOrigin
        """
        includedSites = set([siteOrigin])
        alreadySearched = set([])
        newNeighbors = set([])
        
        for i in range(N):
            for site in (includedSites - alreadySearched):
                newNeighbors = newNeighbors.union(set(self.nearestNeighbors[site])-set([-1]))
            alreadySearched = alreadySearched.union(includedSites - alreadySearched)
            includedSites = includedSites.union(newNeighbors)
            newNeighbors.clear()
        
        return list(includedSites)
    
    def siteProjector(self,siteIndex):
        """Returns a projection operator onto states at siteIndex
        """
        proj = np.zeros((self.hilbertSpaceDimension,self.hilbertSpaceDimension),dtype=complex)
        startIndex = self.stateIndex(siteIndex)
        endIndex = startIndex + self.onsiteDimension
        proj[startIndex:endIndex,startIndex:endIndex] = np.eye(self.onsiteDimension,dtype=complex)
        return proj
    
    def siteListProjector(self,siteList):
        """Returns a projection operator onto all states for sites included in
        siteList
        """    
        return sum([self.siteProjector(siteIndex) for siteIndex in siteList])
    
    def _sitesWithinUnwrappedDistOfVector(self, vector, dist, siteList):
        """Helper function for 2D chern number code below. Returns the unwrapped distance
        from a vector to all sites in siteList
        """
        return [x for x in siteList if np.linalg.norm(self.sitePositions[x] - vector) < dist ]
    
    def _siteDimStep(self, dimension, siteInd):
        """Helper function to caluclate the appropriate siteIntegerVector from
        a siteIndex
        """
        if dimension == 0: 
            return siteInd % self.dimensionLengths[0]
        subtractor = 0
        for n in range(dimension):
            nIndex = self._siteDimStep(n, siteInd)
            nPowSteps = 1
            for i in range(n):
                nPowSteps *= self.dimensionLengths[i]
            subtractor += (nIndex*nPowSteps)
        dimPowSteps = 1
        for i in range(dimension+1):
            dimPowSteps *= self.dimensionLengths[i]
        modIndex = siteInd % dimPowSteps
        stepInd = modIndex - subtractor
        stepInd = ((stepInd*self.dimensionLengths[dimension])/dimPowSteps)
        return stepInd
    
    def _findNearestNeighbors(self,siteIndex):
        """Helper function to find the site indices of the nearest neighbors of
        siteIndex. Considers periodic boundary conditions, if the nearest 
        neighbor doesn't exist (due to boundary or other defect), 
        the function returns -1 at the position of misisng neighbor. 
        The function returns a list of 2x the spatial dimension ordered with 
        the neighbors in the following positions: [+x,-x, +y, -y, ...]
        """
        neighborlist = []
        siteIntVector = self.siteIntegerVector(siteIndex)
        for dimension in range(self.spatialDimension):
            plusVector = np.array(siteIntVector,dtype=int)
            minusVector = np.array(siteIntVector,dtype=int)
            
            plusVector[dimension] += 1
            minusVector[dimension] -= 1
            
            plusFlag = False
            minusFlag = False
            
            if (plusVector[dimension] == self.dimensionLengths[dimension]):
                if(self.periodicBoundaryConditions[dimension]):
                    plusVector[dimension] = 0
                else:
                   plusFlag = True
            if(minusVector[dimension] == -1):
               if(self.periodicBoundaryConditions[dimension]):
                   minusVector[dimension] = self.dimensionLengths[dimension] - 1
               else: 
                   minusFlag = True
           
            if(plusFlag):
                neighborlist.append(-1)
            else:
                neighborlist.append(int(self.siteIndex(plusVector)))
            if(minusFlag):
                neighborlist.append(-1)
            else:
                neighborlist.append(int(self.siteIndex(minusVector)))
        return neighborlist
    
    def translationGenerator(self,dim,plusminus):
        """Returns a translation operator by one in dimension dim, uses PBC
        """
        trans = np.zeros((self.hilbertSpaceDimension,self.hilbertSpaceDimension),dtype=complex)
        identity = np.eye(self.onsiteDimension,dtype=complex)
        
        for siteIndex in range(self.numSites):
            neighborIndex = self.getNeighborSiteIndex(siteIndex,dim,plusminus)
            
            fromBeginIndex = self.stateIndex(siteIndex)
            fromEndIndex = fromBeginIndex + self.onsiteDimension
            
            toBeginIndex = self.stateIndex(neighborIndex)
            toEndIndex =  toBeginIndex + self.onsiteDimension
            
            trans[toBeginIndex:toEndIndex,fromBeginIndex:fromEndIndex] = identity
        
        return trans

class OneDimBravaisLattice(BravaisLattice):
    def plot(self):
        """Plots the positions of the Bravais lattice sites for 2D lattices
        """
        plt.scatter(*zip(*self.statePositions))
        plt.show()
    
class TwoDimBravaisLattice(BravaisLattice):
    
    def plot(self):
        """Plots the positions of the Bravais lattice sites for 2D lattices
        """
        plt.scatter(*zip(*self.statePositions))
        plt.show()
        
    def plotState(self,state,scalingfcn = (lambda x : np.log(1.0+x)*0.5)):
        """Plots the magnitude of the state as radius of circles in realspace, scaled by scaling
        """
        fig,ax = plt.subplots(1)
        ax.set_aspect('equal')
        ax.scatter(*zip(*self.statePositions),s=0.5)
        for index, position in enumerate(self.statePositions):
            circ = Circle((position[0],position[1]),float(scalingfcn(np.real(state[index]*state[index].conj()))),zorder=10)
            ax.add_patch(circ)        
        plt.show()
    
    def _anglesOfSitesRelativeToVector(self, vector, siteList):
        """Helper function for 2D chern number code below. Returns the unwrapped angle
        from a vector to all sites in siteList
        """
        disps = [self.sitePositions[x] - vector for x in siteList]
        angles = [np.arctan2(x[0],x[1]) for x in disps]
        correctedAngles = [angle + 2*np.pi if angle < 0 else angle for angle in angles]
        return correctedAngles
        
    def chernNumber2D(self, proj, sneakyShift = np.array([0.1,0.1])):
        """Intended to work with lattices of two spatial dimensions. Returns
        the real space chern number of the hamiltonian with lower flat band bandprojector. 
        """
        dimLengthMin = min(self.dimensionPhysicalLengths)
        maxDistance = dimLengthMin/4.0
        
        middleVector = np.array([i//2 for i in self.dimensionLengths])
        middlePosition = np.array(self.sitePositions[self.siteIndex(middleVector)])
        shiftedMiddle = middlePosition + sneakyShift
        
        consideredSites = self._sitesWithinUnwrappedDistOfVector(shiftedMiddle,maxDistance,range(self.numSites))
        angles = self._anglesOfSitesRelativeToVector(shiftedMiddle,consideredSites)
        
        sitesRegionA  = [s for s,a in zip(consideredSites,angles) if 0 <= a < 2*np.pi/3]
        sitesRegionB  = [s for s,a in zip(consideredSites,angles) if 2*np.pi/3 <= a < 4*np.pi/3]
        sitesRegionC  = [s for s,a in zip(consideredSites,angles) if 4*np.pi/3 <= a < 6*np.pi/3]
        
        mask_A = self.siteListProjector(sitesRegionA)
        mask_B = self.siteListProjector(sitesRegionB)
        mask_C = self.siteListProjector(sitesRegionC)
        radial_mask = self.siteListProjector(consideredSites)
        
        return 12*np.pi*1j*np.trace(proj @ mask_A @ proj @ mask_B @ proj @ mask_C @ proj @ radial_mask 
                                   - proj @ mask_A @ proj @ mask_C @ proj @ mask_B @ proj @ radial_mask )
        
    def displayState(self,state,absoluteScale=False):
        if absoluteScale:
            normfunct = plt.Normalize(0,1)
        else:
            normfunct = None
        
        shapedState = state.reshape(self.onsiteDimension*self.dimensionLengths[0],self.dimensionLengths[1]) #this will always work but is ugly
        out, subs = plt.subplots(2)
        subs[0].set_title("State Maginitude")
        subs[0] = plt.imshow(np.absolute(shapedState),cmap = plt.cm.Reds, norm = normfunct)
        subs[1].set_title("State Complex Arg")
        subs[1] = plt.imshow(np.angle(shapedState),cmap = plt.cm.hsv, norm=plt.Normalize(0,2*np.pi))

### Below are implemented lattices for convineince
        
def oneDimensionalLattice(L,periodicBoundaryConditions=[],a=1.0):
    """Returns a one dimenisonal lattice of length L
    """
    eX = np.array([a])
    return OneDimBravaisLattice(np.matrix([eX]),[L],periodicBoundaryConditions=periodicBoundaryConditions)    

def oneDimensionalBipartiteLattice(L,periodicBoundaryConditions=[],a=1.0):
    """Returns a one dimenisonal lattice of length L
    """
    eX = np.array([a])
    unitcellBasis = np.array([[0.0],[a/2.0]])
    return OneDimBravaisLattice(np.matrix([eX]),[L],unitcellBasis,periodicBoundaryConditions)
    
def rectangularLattice(W,H,periodicBoundaryConditions=[],a=1.0,b=1.0):
    """Returns a rectangular lattice of x-length W and y-length H, 
    unit cell area = a x b
    """
    eX = np.array([a,0])
    eY = np.array([0,b])
    return TwoDimBravaisLattice(np.matrix([eX,eY]),[W,H],np.array([[0.0,0.0]]),periodicBoundaryConditions)

def checkerboardLattice(W,H,periodicBoundaryConditions=[],a=1.0):
    """Returns a checkerboard square lattice with unitcell area axa
    """
    eX = np.array([a,0])
    eY = np.array([0,a])
    unitcellBasis = np.array([[0.0,0.0],[a/2.0,a/2.0]])
    return TwoDimBravaisLattice(np.matrix([eX,eY]),[W,H],unitcellBasis,periodicBoundaryConditions)

def unitTriangularLattice(W,H,periodicBoundaryConditions=[],a=1.0):
    """Returns a equilataral triangular lattice of width W and height H, 
    unit cell side a
    """
    eX = np.array([a,0])
    eY = np.array([0.5*a,0.5*a*sqrt(3.0)])
    return TwoDimBravaisLattice(np.matrix([eX,eY]),[W,H],np.array([[0.0,0.0]]),periodicBoundaryConditions)

def honeycombLattice(W,H,periodicBoundaryConditions=[],a=1.0):
    """Returns a regular honeycomb lattice with width W and height H and 
    spacing between onsite orbitals a
    """
    eX = np.array([3.0*sqrt(3.0)*a/2.0,3.0*a/2])
    eY = np.array([3.0*sqrt(3.0)*a/2.0,-3.0*a/2])
    unitcellBasis = np.array([[0.0,0.0],[a,0.0]])
    return TwoDimBravaisLattice(np.matrix([eX,eY]),[W,H],
                          unitcellBasis,periodicBoundaryConditions)
    
    