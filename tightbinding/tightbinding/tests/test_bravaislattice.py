# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:12:00 2018

@author: dominic
"""
import numpy as np
from unittest import TestCase

import tightbinding.bravaislattice as bl

class TestBravaisLattice(TestCase):
    def test_siteIndex(self):
        lattice = bl.rectangularLattice(3,5)
        siteVector = np.array([1,3])
        expectedIndex = 10
        self.assertTrue(lattice.siteIndex(siteVector)==expectedIndex)
        
        lattice = bl.rectangularLattice(4,6)
        siteVector = np.array([1,3])
        expectedIndex = 13
        self.assertTrue(lattice.siteIndex(siteVector) == expectedIndex)
        
        lattice = bl.rectangularLattice(4,3)
        siteVector = np.array([1,2])
        expectedIndex = 9
        self.assertTrue(lattice.siteIndex(siteVector) == expectedIndex)
        
        lattice = bl.rectangularLattice(3,4)
        siteVector = np.array([1,2])
        expectedIndex = 7
        self.assertTrue(lattice.siteIndex(siteVector) == expectedIndex)
        
    def test_siteIntegerVector(self):
        lattice = bl.rectangularLattice(3,5)
        siteIndex = 8
        expectedVector = np.array([2,2])
        self.assertTrue(np.all(lattice.siteIntegerVector(siteIndex)==expectedVector))
        
        lattice = bl.rectangularLattice(5,3)
        siteIndex = 8
        expectedVector = np.array([3,1])
        self.assertTrue(np.all(lattice.siteIntegerVector(siteIndex)==expectedVector))
        
        lattice = bl.rectangularLattice(2,4)
        siteIndex = 5
        expectedVector = np.array([1,2])
        self.assertTrue(np.all(lattice.siteIntegerVector(siteIndex)==expectedVector))
        
        lattice = bl.rectangularLattice(4,2)
        siteIndex = 5
        expectedVector = np.array([1,1])
        self.assertTrue(np.all(lattice.siteIntegerVector(siteIndex)==expectedVector))
        
        lattice = bl.rectangularLattice(3,4)
        siteIndex = 5
        expectedVector = np.array([2,1])
        self.assertTrue(np.all(lattice.siteIntegerVector(siteIndex)==expectedVector))
        
        lattice = bl.rectangularLattice(4,3)
        siteIndex = 5
        expectedVector = np.array([1,1])
        self.assertTrue(np.all(lattice.siteIntegerVector(siteIndex)==expectedVector))
        
        
    def test_stateIndex(self):
        lattice = bl.honeycombLattice(3,5)
        siteIndex = 3
        expectedStateIndex0 = 6
        expectedStateIndex1 = 7
        self.assertTrue(lattice.stateIndex(siteIndex,0) == expectedStateIndex0)
        self.assertTrue(lattice.stateIndex(siteIndex,1) == expectedStateIndex1)
        
        lattice = bl.honeycombLattice(3,4)
        siteIndex = 3
        expectedStateIndex0 = 6
        expectedStateIndex1 = 7
        self.assertTrue(lattice.stateIndex(siteIndex,0) == expectedStateIndex0)
        self.assertTrue(lattice.stateIndex(siteIndex,1) == expectedStateIndex1)
        
        lattice = bl.honeycombLattice(4,6)
        siteIndex = 3
        expectedStateIndex0 = 6
        expectedStateIndex1 = 7
        self.assertTrue(lattice.stateIndex(siteIndex,0) == expectedStateIndex0)
        self.assertTrue(lattice.stateIndex(siteIndex,1) == expectedStateIndex1)
        
    def test_siteAndOrbitalIndex(self):
        lattice = bl.honeycombLattice(7,3)
        stateIndex = 13
        siteInd, orbInd = lattice.siteAndOrbitalIndex(stateIndex)
        expectedSiteInd = 6
        expectedOrbInd = 1
        self.assertTrue(siteInd == expectedSiteInd)
        self.assertTrue(orbInd == expectedOrbInd)
        
        lattice = bl.honeycombLattice(7,3)
        stateIndex = 12
        siteInd, orbInd = lattice.siteAndOrbitalIndex(stateIndex)
        expectedSiteInd = 6
        expectedOrbInd = 0
        self.assertTrue(siteInd == expectedSiteInd)
        self.assertTrue(orbInd == expectedOrbInd)
        
        lattice = bl.honeycombLattice(4,3)
        stateIndex = 13
        siteInd, orbInd = lattice.siteAndOrbitalIndex(stateIndex)
        expectedSiteInd = 6
        expectedOrbInd = 1
        self.assertTrue(siteInd == expectedSiteInd)
        self.assertTrue(orbInd == expectedOrbInd)
        
        lattice = bl.honeycombLattice(4,3)
        stateIndex = 12
        siteInd, orbInd = lattice.siteAndOrbitalIndex(stateIndex)
        expectedSiteInd = 6
        expectedOrbInd = 0
        self.assertTrue(siteInd == expectedSiteInd)
        self.assertTrue(orbInd == expectedOrbInd)
        
        lattice = bl.honeycombLattice(4,6)
        stateIndex = 13
        siteInd, orbInd = lattice.siteAndOrbitalIndex(stateIndex)
        expectedSiteInd = 6
        expectedOrbInd = 1
        self.assertTrue(siteInd == expectedSiteInd)
        self.assertTrue(orbInd == expectedOrbInd)
        
        lattice = bl.honeycombLattice(4,6)
        stateIndex = 12
        siteInd, orbInd = lattice.siteAndOrbitalIndex(stateIndex)
        expectedSiteInd = 6
        expectedOrbInd = 0
        self.assertTrue(siteInd == expectedSiteInd)
        self.assertTrue(orbInd == expectedOrbInd)
        
    def test_getNeighborSiteIndex(self):
        lattice_pbc_both = bl.rectangularLattice(3,5)
        lattice_pbcx_npbcy = bl.rectangularLattice(3,5,[True,False])
        lattice_npbcx_pbcy = bl.rectangularLattice(3,5,[False,True])
        lattice_npbc_both = bl.rectangularLattice(3,5,[False,False])
        
        self.assertTrue(lattice_pbc_both.getNeighborSiteIndex(8,0,1) == 6)
        self.assertTrue(lattice_pbcx_npbcy.getNeighborSiteIndex(8,0,1) == 6)
        self.assertTrue(lattice_npbcx_pbcy.getNeighborSiteIndex(8,0,1) == -1)
        self.assertTrue(lattice_npbc_both.getNeighborSiteIndex(8,0,1) == -1)
        
        self.assertTrue(lattice_pbc_both.getNeighborSiteIndex(6,0,-1) == 8)
        self.assertTrue(lattice_pbcx_npbcy.getNeighborSiteIndex(6,0,-1) == 8)
        self.assertTrue(lattice_npbcx_pbcy.getNeighborSiteIndex(6,0,-1) == -1)
        self.assertTrue(lattice_npbc_both.getNeighborSiteIndex(6,0,-1) == -1)
        
        self.assertTrue(lattice_pbc_both.getNeighborSiteIndex(7,0,-1) == 6)
        self.assertTrue(lattice_pbcx_npbcy.getNeighborSiteIndex(7,0,-1) == 6)
        self.assertTrue(lattice_npbcx_pbcy.getNeighborSiteIndex(7,0,-1) == 6)
        self.assertTrue(lattice_npbc_both.getNeighborSiteIndex(7,0,-1) == 6)
        
        self.assertTrue(lattice_pbc_both.getNeighborSiteIndex(7,0,1) == 8)
        self.assertTrue(lattice_pbcx_npbcy.getNeighborSiteIndex(7,0,1) == 8)
        self.assertTrue(lattice_npbcx_pbcy.getNeighborSiteIndex(7,0,1) == 8)
        self.assertTrue(lattice_npbc_both.getNeighborSiteIndex(7,0,1) == 8)
        
        self.assertTrue(lattice_pbc_both.getNeighborSiteIndex(13,1,1) == 1)
        self.assertTrue(lattice_pbcx_npbcy.getNeighborSiteIndex(13,1,1) == -1)
        self.assertTrue(lattice_npbcx_pbcy.getNeighborSiteIndex(13,1,1) == 1)
        self.assertTrue(lattice_npbc_both.getNeighborSiteIndex(13,1,1) == -1)
        
        self.assertTrue(lattice_pbc_both.getNeighborSiteIndex(1,1,-1) == 13)
        self.assertTrue(lattice_pbcx_npbcy.getNeighborSiteIndex(1,1,-1) == -1)
        self.assertTrue(lattice_npbcx_pbcy.getNeighborSiteIndex(1,1,-1) == 13)
        self.assertTrue(lattice_npbc_both.getNeighborSiteIndex(1,1,-1) == -1)
        
        self.assertTrue(lattice_pbc_both.getNeighborSiteIndex(10,1,1) == 13)
        self.assertTrue(lattice_pbcx_npbcy.getNeighborSiteIndex(10,1,1) == 13)
        self.assertTrue(lattice_npbcx_pbcy.getNeighborSiteIndex(10,1,1) == 13)
        self.assertTrue(lattice_npbc_both.getNeighborSiteIndex(10,1,1) == 13)
        
        self.assertTrue(lattice_pbc_both.getNeighborSiteIndex(10,1,-1) == 7)
        self.assertTrue(lattice_pbcx_npbcy.getNeighborSiteIndex(10,1,-1) == 7)
        self.assertTrue(lattice_npbcx_pbcy.getNeighborSiteIndex(10,1,-1) == 7)
        self.assertTrue(lattice_npbc_both.getNeighborSiteIndex(10,1,-1) == 7)
        
    def test_getRelativeSiteIndex(self):
        lattice_pbc_both = bl.rectangularLattice(3,5)
        lattice_npbc_both = bl.rectangularLattice(3,5,[False,False])
        self.assertTrue(lattice_pbc_both.getRelativeSiteIndex(10,np.array([2,2],dtype=int))==0)
        self.assertTrue(lattice_npbc_both.getRelativeSiteIndex(10,np.array([2,2],dtype=int))==-1)
        self.assertTrue(lattice_pbc_both.getRelativeSiteIndex(10,np.array([1,-1],dtype=int))==8)
        self.assertTrue(lattice_npbc_both.getRelativeSiteIndex(10,np.array([1,-1],dtype=int))==8)
        self.assertTrue(lattice_pbc_both.getRelativeSiteIndex(10,np.array([-2,-2],dtype=int))==5)
        self.assertTrue(lattice_npbc_both.getRelativeSiteIndex(10,np.array([-2,-2],dtype=int))==-1)
    
    def test_sitesWithinNSteps(self):
        lattice_pbc_both = bl.rectangularLattice(3,5)
        lattice_npbc_both = bl.rectangularLattice(3,5,[False,False])
        self.assertTrue(set(lattice_pbc_both.sitesWithinNSteps(0,0)) == set([0]))
        self.assertTrue(set(lattice_npbc_both.sitesWithinNSteps(0,0)) == set([0]))
        self.assertTrue(set(lattice_pbc_both.sitesWithinNSteps(0,1)) == set([0,1,2,3,12]))
        self.assertTrue(set(lattice_npbc_both.sitesWithinNSteps(0,1)) == set([0,1,3]))
        self.assertTrue(set(lattice_pbc_both.sitesWithinNSteps(0,2)) == set([0,1,2,3,4,5,6,9,12,13,14]))
        self.assertTrue(set(lattice_npbc_both.sitesWithinNSteps(0,2)) == set([0,1,2,3,4,6]))
    
    def test_nearestNeighbors(self):
        lattice_pbc_both = bl.rectangularLattice(3,5)
        lattice_npbc_both = bl.rectangularLattice(3,5,[False,False])
        
        self.assertTrue(lattice_pbc_both.nearestNeighbors[0] == [1,2,3,12])
        self.assertTrue(lattice_npbc_both.nearestNeighbors[0] == [1,-1,3,-1])
        self.assertTrue(lattice_pbc_both.nearestNeighbors[14] == [12,13,2,11])
        self.assertTrue(lattice_npbc_both.nearestNeighbors[14] == [-1,13,-1,11])
        
        