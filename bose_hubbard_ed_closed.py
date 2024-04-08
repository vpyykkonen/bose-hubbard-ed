# Code for exact diagonalization of Bose-Hubbard lattice models 
import os
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from numpy import linalg
from quspin.operators.hamiltonian_core import commutator

import sys

import matplotlib as mpl


from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # bosonic Hilbert space
from quspin.tools.measurements import diag_ensemble # diagonal ensemble
import numpy as np # general math functions
import math
import matplotlib.pyplot as plt # plotting library
import matplotlib.colors as colors
from matplotlib import cm

import h5py

SMALL_SIZE = 20
MEDIUM_SIZE = 23
BIGGER_SIZE = 25

mpl.rcParams['axes.titlesize'] = SMALL_SIZE     # fontsize of the axes title
mpl.rcParams['axes.labelsize'] = MEDIUM_SIZE    # fontsize of the x and y labels
mpl.rcParams['xtick.labelsize'] = SMALL_SIZE    # fontsize of the tick labels
mpl.rcParams['ytick.labelsize'] = SMALL_SIZE    # fontsize of the tick labels
mpl.rcParams['legend.fontsize'] = SMALL_SIZE    # legend fontsize
mpl.rcParams['figure.titlesize'] = BIGGER_SIZE  # fontsize of the figure title
mpl.rcParams['figure.constrained_layout.use'] = True

import distinctipy

import cmasher as cmr
import colorcet as cc


# Generates states with Nb bosons in a single-particle state defined in amps
# as a list of amplitudes on single-particle basis states
# Returns a list of states with various single-particle basis states occupations 'states' and their respective ampltiudes 'states_weight'
# E.g. if the single-particle Hilbert space has dimension 3
# and one considers the single-particle state amps=[1,1]/np.sqrt(2),
# the corresponding result with Nb=2 is states = [[2,0],[1,1],[0,2]]
# and states_weight = [1/2,1/np.sqrt(2),1/2]
def generate_states(Nb,amps):
    states = []
    states_weight = []
    if(len(amps) == 1):
        return [str(Nb)],[amps[0]**Nb]

    for nb in range(Nb+1):
        state_str = str(nb)
        state_weight = amps[0]**nb*np.sqrt(math.factorial(Nb)/(math.factorial(nb)*math.factorial(Nb-nb)))
        states_prev,states_weight_prev = generate_states(Nb-nb,amps[1:])
        count = 0
        for state in states_prev:
            states.append(state_str+state)
            states_weight.append(states_weight_prev[count]*state_weight)
            count += 1

    return states,states_weight

##### define model parameters
# initial seed for random number generator
np.random.seed(0) # seed is 0 to produce plots from QuSpin2 paper
# setting up parameters of simulation

# model-independent parameters
Nb = 2 # number of bosons
sps = Nb+1 # number of states per site
U = 1.0 # interaction strength
if Nb == 1:
    U = 0

# Simulation time duration in units of inverse energy (hbar=1)
start,stop,num = 0,1000,2001 

# Give lattice name and the type of the initial state (defined below)
lattice_name = "three_site"
init_type = "edge_site_left"

# Get commandline options if available
if len(sys.argv) > 1:
    lattice_name = sys.argv[1]
    init_type = sys.argv[2]
    Nb = int(sys.argv[3])
    U = float(sys.argv[4])
    if Nb == 1:
        U = 0
    start = float(sys.argv[5])
    stop = float(sys.argv[6])
    num = int(sys.argv[7])

##### set up Hamiltonian and observables #####
# Insert a new model by defining an if block with the specified lattice name
# following the example below
if lattice_name == "three_site":
    # Model has three sites A, B and C with hoppings tAB, tAC, and tBC.
    # The model has a localized single-particle eigenstate on sites A and B if tAB = 0
    # rAB sets the amplitude ratio between the A and B sites for this state
    #     B
    #     /\
    #    /__\
    #   A    C
    rAB = -4  # rAB = -tBC/tAC
    if len(sys.argv) > 8:
        rAB = float(sys.argv[8])

    epsC = 0.0
    tAC = 1.0
    tAB = 0.0
    tBC = -rAB*tAC
    #name = "three_site_"+str(rAB)+"_tAB_"+str(tAB)+"_"
    name = "three_site_"+str(rAB)+"_"
    
    #tAB = epsA/(-1./rAB+rAB)
    
    #epsA = tAB*(tBC/tAC-tAC/tBC)
    epsA = 0.0
    
    VB = epsA
    n_sites = 3
    
    hop_list = [[-tAB,0,1],[-tAC,0,2],[-tBC,1,2]]
    hop_list_hc = [[J.conjugate(),j,i] for J,i,j in hop_list] # add h.c. terms
    
    E_list = [[epsA,0],[epsC,2]] 
    int_list = [[U/2,i,i,i,i] for i in range(n_sites)]
    basis = boson_basis_1d(n_sites,Nb=Nb) 
    psi0 = np.zeros(basis.Ns,dtype=np.complex128)
    
    if init_type == "edge_state_left":
        n_zeros_left = 0
        n_zeros_right = n_sites-2
        amps = np.array([rAB,1.0],dtype=complex)
        amps /= np.linalg.norm(amps)
        states,weights = generate_states(Nb,amps)
        for n in range(len(states)):
            state = n_zeros_left*"0"+states[n]+n_zeros_right*"0"
            psi0[basis.index(state)] = weights[n]
    
    elif init_type == "edge_site_left":
        n_zeros_left = 0
        n_zeros_right = n_sites-1
        states,weights = generate_states(Nb,[1.0])
        for n in range(len(states)):
            state = n_zeros_left*"0"+states[n]+n_zeros_right*"0"
            print(state)
            psi0[basis.index(state)] = weights[n]
    else:
        print("Invalid initial state given. Exiting.")
        exit()

#sawtooth lattice
elif lattice_name == "sawtooth":
    # Sawtooth lattice is a one-dimensional lattice with two sites per unit cell,
    # formed in a chain of triangles or sawtooths as
    #      B   B   B
    #      /\  /\  /\
    #  .../__\/__\/__\...
    #     A   A   A   A
    # where at each vertex is one site, labeled A or B.
    # If tAB = sqrt(2)*tABB, the system has localized state in the 'Vs'
    # 1    1
    #  \  /
    #   \/
    # sqrt(2)

    # If also VB = -tAA, then system has localized states at the edges
    # sqrt(2) and sqrt(2)
    #   /           \
    #  /             \
    # -1             -1
    #



    n_cells = 2 # number of triangles
    VB = -1.0*t_AA 	 # boundary potential at left and right edge A sites
    t_AA = -1.0  # AA hopping
    t_AB = np.sqrt(2)*t_AA # AB hopping

    if len(sys.argv)>8:
        n_cells = int(sys.argv[8])
        VB = float(sys.argv[9])

    name = "sawtooth"+str(n_cells)
    n_sites = 2*n_cells+1
    
    hop_list = [[t_AA,i,(i+2)] for i in range(0,n_sites-2,2)] # AA hopping
    hop_list.extend([[t_AB,i,(i+1)] for i in range(0,n_sites-1,1)]) # AB hopping
    hop_list_hc = [[J.conjugate(),j,i] for J,i,j in hop_list] # add h.c. terms
    E_list = [[VB,0],[VB,n_sites-1]] # boundary potential
    int_list = [[U/2,i,i,i,i] for i in range(n_sites)]
    
    basis = boson_basis_1d(n_sites,Nb=Nb) 
    
    psi0 = np.zeros(basis.Ns,dtype=np.complex128)
    if init_type == "edge_state_left":
        n_zeros_left = 0
        n_zeros_right = n_sites-2
        states,weights = generate_states(Nb,[np.sqrt(2.0/3.0),-1.0/np.sqrt(3.0)])
        #states,weights = generate_states(Nb,[1.0])
        for n in range(len(states)):
            state = n_zeros_left*"0"+states[n]+n_zeros_right*"0"
            psi0[basis.index(state)] = weights[n]

    elif init_type == "edge_site_left":
        n_zeros_left = 0
        n_zeros_right = n_sites-1
        states,weights = generate_states(Nb,[1.0])
        for n in range(len(states)):
            state = n_zeros_left*"0"+states[n]+n_zeros_right*"0"
            psi0[basis.index(state)] = weights[n]
    else:
        print("Invalid initial state given. Exiting.")
        exit()
    
elif lattice_name == "diamond":
    # 1D chain of connected diamonds of the form
    #     B
    #     /\  /\
    #...A/  \/  \...
    #    \  /\  /
    #     \/  \/
    #     C
    # Has localized state if a pi flux is instered. 
    # Flux can be realized e.g. as done below
    n_cells = 2
    flux = np.pi
    if len(sys.argv)>8:
        n_cells = int(sys.argv[8])
        flux = float(sys.argv[9])
    t = 1.0
    if len(sys.argv)>10:
        t = float(sys.argv[10])
    name = "diamond"+str(n_cells)+"_flux"+str(flux)+"_"+str(t)+"_"
    VB = 0.0
    n_sites = 3*n_cells+1

    hop_list = [[-t,i,i+1] for i in range(0,n_sites-2,3)] # a_n b_n
    hop_list.extend([[-t,i,i+2] for i in range(0,n_sites-2,3)]) # a_n c_n
    hop_list.extend([[-t*np.exp(-1.0j*flux),i+1,i+3] for i in range(0,n_sites-2,3)]) # b_{n}, a_{n+1}
    hop_list.extend([[-t,i+2,i+3] for i in range(0,n_sites-2,3)]) # c_{n}, a_{n+1}
    hop_list_hc = [[J.conjugate(),j,i] for J,i,j in hop_list] # add h.c.
    E_list = []
    int_list = [[U/2,i,i,i,i] for i in range(n_sites)]
    basis = boson_basis_1d(n_sites,Nb=Nb) 
    
    psi0 = np.zeros(basis.Ns,dtype=np.complex128)
    if init_type == "edge_state_left":
        amps = np.asarray([np.sqrt(2),1.0,1.0])
        amps /= np.linalg.norm(amps)
        n_zeros_left = 0
        n_zeros_right = n_sites-3
        states, weights = generate_states(Nb,amps)
        for n in range(len(states)):
            state = n_zeros_left*"0"+states[n]+n_zeros_right*"0"
            psi0[basis.index(state)] = weights[n]
    elif init_type == "edge_site_left":
        n_zeros_left = 0
        n_zeros_right = n_sites-1
        states,weights = generate_states(Nb,[1.0])
        for n in range(len(states)):
            state = n_zeros_left*"0"+states[n]+n_zeros_right*"0"
            print(state)
            psi0[basis.index(state)] = weights[n]
    else:
        print("Invalid initial state given. Exiting.")
        exit()
else:
    print("Invalid lattice name given. Exiting.")
    exit()

# set up static and dynamic lists
static = [
			["+-",hop_list],    # hopping
			["+-",hop_list_hc], # hopping h.c.
                        ["n",E_list],
                        ["++--",int_list]
		]
dynamic = [] # no dynamic operators

# build real-space Hamiltonian
H = hamiltonian(static,dynamic,basis=basis,dtype=np.complex128,check_symm=False)

state_idxs = []
state_labels = []
state_ints = []

for state in basis.states:
    state_idxs.append(basis.index(state))
    state_labels.append(basis.int_to_state(state))
    state_ints.append(state)

state_idxs = np.asarray(state_idxs)
state_ints = np.asarray(state_ints)

no_checks = dict(check_herm=False,check_symm=False,check_pcon=False)

# setting up observables
# site occupation numbers
n_list = [hamiltonian([["n",[[1.0,i]]]],[],basis=basis,dtype=np.complex128,**no_checks) for i in range(n_sites)]

ts = np.linspace(start,stop,num)

evo = H.evolve(psi0, 0, ts)
ns = np.zeros([n_sites,num])
for n in range(n_sites):
    ns[n,:] = n_list[n].expt_value(evo).real

E,V=H.eigh() # Calculate Hamiltonian eigenvalues and eigenvectors

# Calculate overlaps between initial state and the eigenvectors
overlaps = np.dot(np.transpose(np.conj(V)),psi0)

output_name = "closed"+\
        "_"+init_type+\
        "_Nb"+str(Nb)+\
        "_U{:.4f}".format(U)

file = h5py.File('./Data/'+name+output_name+'.h5','w')
file.create_dataset('lattice_name',data=lattice_name,dtype=h5py.string_dtype(encoding='utf-8'))
file.create_dataset('init_type',data=init_type,dtype=h5py.string_dtype(encoding='utf-8'))
file.create_dataset('U',data=U)
if lattice_name == "three_site":
    file.create_dataset('rAB',data=rAB)
if lattice_name == "diamond":
    file.create_dataset('flux',data=flux)
    file.create_dataset('n_cells',data=n_cells)
if lattice_name == "sawtooth":
    file.create_dataset('n_cells',data=n_cells)

file.create_dataset('eigenvalues',data=E)
file.create_dataset('eigenvectors_r',data=np.real(V))
file.create_dataset('eigenvectors_i',data=np.imag(V))
file.create_dataset('overlaps_r',data=np.real(overlaps))
file.create_dataset('overlaps_i',data=np.imag(overlaps))
file.create_dataset('n_sites',data=n_sites)
file.create_dataset('Nb',data=Nb)
file.create_dataset('ts',data=ts)
file.create_dataset('psi_t_r',data=np.real(evo))
file.create_dataset('psi_t_i',data=np.imag(evo))
file.create_dataset('ns_t',data=np.real(ns))
file.create_dataset('state_ints',data=state_ints,dtype='i8')
file.create_dataset('state_idxs',data=state_idxs,dtype='i8')
file.create_dataset('state_labels',data=state_labels,dtype=h5py.string_dtype(encoding='utf-8'))
file.close()
print("Data saved at: " + "./Data/"+name+output_name+".h5")

