# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 12:10:16 2018

@author: dominic
"""

import numpy as np
from numpy import linalg as LA

def pauliX():
    """Returns sigmaX
    """
    return np.array([[0,1.0],[1.0,0]],dtype=complex)

def pauliY(): 
    """Returns sigmaY
    """
    return np.array([[0,-1j],[1j,0]],dtype=complex)

def pauliZ():
    """Returns sigmaZ
    """
    return np.array([[1,0],[0,-1]],dtype=complex)

def pauliD():
    print("Welcome to the Jersey Shore.")
    return
                   
def projector(evec):
    """Returns the projector onto state evec
    """
    nevec = evec/np.inner(evec,evec.conj())
    return np.outer(nevec,nevec.conj())

def flatten(hamiltonian, gap=0, w = [], v = []):
    """Returns the flattened hamiltonian associated with hamiltonian, with
    eigenvalues one for states above gap and eigenvalues minus one for states
    below gap, if the eigenvalues and eigenvectors are known, one may pass them
    into arguments w and v to avoid repeating diagonalization
    """
    if(len(w) == 0):
        w, v = LA.eigh(hamiltonian)       
    signedProjectors = map(lambda x,y: np.sign(x-gap)*projector(y), w, v.T)
    return sum(signedProjectors)

def flattenUpper(hamiltonian,gap=0, w = [], v = []):
    """Returns +P where P is the projector onto eigenstates of the hamiltonain
    with eigenvalues larger than gap
    """
    if(len(w) == 0):
        w,v = LA.eigh(hamiltonian)
    proj = np.zeros(np.shape(hamiltonian),dtype=complex)
    for index in range(len(w)):
        if np.sign(w[index] - gap) > 0:
            proj += projector(v[:,index])
    return proj

def flattenLower(hamiltonian, gap = 0, w = [], v = []):
    """Returns -P where P is the projector onto eigenstates of the hamiltonain
    with eigenvalues smaller than gap
    """
    if(len(w) == 0):
        w,v = LA.eigh(hamiltonian)
    
    proj = np.zeros(np.shape(hamiltonian),dtype=complex)
    for index in range(len(w)):
        if np.sign(w[index] - gap) < 0:
            proj += projector(v[:,index])
    return proj

def transformHamiltonian(hamiltonian,unitary):
    """Returns hamiltonian evolved by unitary
    """
    return np.dot(np.dot(unitary,hamiltonian),unitary.conj().T)

def commutator(h1,h2):
    """Returns the commutator of h1 and h2
    """
    return np.dot(h1,h2) - np.dot(h2,h1)

def lowerBandwidth(hamiltonian, gap=0, w=[]):
    """Calculates the bandwidth of the band below gap, if the eigenvalues are 
    known one may pass them as w to avoid recomputation
    """
    if(len(w) == 0): 
        w = LA.eigvalsh(hamiltonian)
    lowerEigvals = [eigval for eigval in w if eigval < gap]
    return (max(lowerEigvals) - min(lowerEigvals))

def upperBandwidth(hamiltonian, gap=0, w=[]):
    """Calculates the bandwidth of the band above gap, if the eigenvalues are 
    known one may pass them as w to avoid recomputation
    """
    if(len(w) == 0): 
        w = LA.eigvalsh(hamiltonian)
    upperEigvals = [eigval for eigval in w if eigval > gap]
    return (max(upperEigvals) - min(upperEigvals))

def lowerFlatnessRatio(hamiltonian, gap=0, w=[]):
    """Calculates the flatness ratio of the band of hamiltonian below gap, 
    where a ratio closer to 0 indicates more flat, if the eigenvalues are known
    they may be passed as w to avoid recomputation
    """
    if(len(w) == 0):
        w = LA.eigvalsh(hamiltonian)
        
    lowerEigvals = [eigval for eigval in w if eigval < gap]
    upperEigvals = [eigval for eigval in w if eigval > gap]
    
    bandgap = min(upperEigvals) - max(lowerEigvals)
    bandwidthLower = max(lowerEigvals) - min(lowerEigvals)
    
    return bandwidthLower/bandgap

def upperFlatnessRatio(hamiltonian, gap=0, w=[]):
    """Calculates the flatness ratio of the band of hamiltonian above gap, 
    where a ratio closer to 0 indicates more flat, if the eigenvalues are known
    they may be passed as w to avoid recomputation
    """
    if(len(w) == 0):
        w = LA.eigvalsh(hamiltonian)
        
    lowerEigvals = [eigval for eigval in w if eigval < gap]
    upperEigvals = [eigval for eigval in w if eigval > gap]
    
    bandgap = min(upperEigvals) - max(lowerEigvals)
    bandwidthUpper = max(upperEigvals) - min(upperEigvals)
    
    return bandwidthUpper/bandgap
    
def flatnessRatio(hamiltonian, gap=0, w=[]):
    """Computes the maximum of the lower and upper flatness ratios
    """
    if(len(w) == 0):
        w = LA.eigvalsh(hamiltonian)
    lowerEigvals = [eigval for eigval in w if eigval < gap]
    upperEigvals = [eigval for eigval in w if eigval > gap]
    
    bandgap = min(upperEigvals) - max(lowerEigvals)
    
    bandwidthLower = max(lowerEigvals) - min(lowerEigvals)
    bandwidthUpper = max(upperEigvals) - min(upperEigvals)
    
    bandwidth = max([bandwidthLower,bandwidthUpper])
    
    return bandwidth/bandgap

    
        
    
    