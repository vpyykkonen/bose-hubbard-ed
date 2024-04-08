from __future__ import print_function, division
#
import sys,os
#os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='6' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='6' # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
from quspin.operators import hamiltonian, commutator, anti_commutator
from quspin.basis import boson_basis_1d # Hilbert space spin basis_1d
from quspin.tools.evolution import evolve
from quspin.tools.misc import get_matvec_function
import numpy as np
from six import iteritems # loop over elements of dictionary
import matplotlib.pyplot as plt # plotting library

from matplotlib import colors
#
import math
import itertools

import cmasher as cmr

import matplotlib as mpl


import h5py


import cmasher as cmr
import colorcet as cc

import distinctipy

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

# Gives list of state indices 
def n_particle_indices(basis,n,n_sites):
    indices = []
    states, weights = generate_states(n,np.ones([n_sites],dtype=complex))
    for state in states:
        indices.append(basis.index(state))
    return indices


###### model parameters
Nb_max = 5
Nb = [n for n in range(Nb_max+1)]

lattice_name = "three_site"
init_type = "edge_site_left"

gammaL1 = 0.0 # loss at left edge
gammaL2 = 0.0 # gain at left edge
gammaR1 = 0.1 # loss at right edge
gammaR2 = 0.0 # gain at right edge

#Interaction strength
U = 1
if Nb_max == 1 or Nb_max == 0:
    U = 0

# Time evolution parameters
start,stop,num = 0.1,10,15
ts=np.linspace(start,stop,num)

if len(sys.argv) > 1:
    lattice_name = sys.argv[1]
    init_type = sys.argv[2]
    if init_type != "edge_state_left" and init_type != "edge_site_left":
        print("Wrong initial state given. Exiting.")
        exit()
    Nb_max = int(sys.argv[3])
    Nb = [n for n in range(Nb_max+1)]
    U = float(sys.argv[4])
    if Nb_max == 1:
        U = 0.0
    gammaR1 = float(sys.argv[5])
    start = float(sys.argv[6])
    stop = float(sys.argv[7])
    num = int(sys.argv[8])

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
        states,weights = generate_states(Nb_max,amps)
        for n in range(len(states)):
            state = n_zeros_left*"0"+states[n]+n_zeros_right*"0"
            psi0[basis.index(state)] = weights[n]
    
    elif init_type == "edge_site_left":
        n_zeros_left = 0
        n_zeros_right = n_sites-1
        states,weights = generate_states(Nb_max,[1.0])
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
        states,weights = generate_states(Nb_max,[np.sqrt(2.0/3.0),-1.0/np.sqrt(3.0)])
        #states,weights = generate_states(Nb,[1.0])
        for n in range(len(states)):
            state = n_zeros_left*"0"+states[n]+n_zeros_right*"0"
            psi0[basis.index(state)] = weights[n]

    elif init_type == "edge_site_left":
        n_zeros_left = 0
        n_zeros_right = n_sites-1
        states,weights = generate_states(Nb_max,[1.0])
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
        states, weights = generate_states(Nb_max,amps)
        for n in range(len(states)):
            state = n_zeros_left*"0"+states[n]+n_zeros_right*"0"
            psi0[basis.index(state)] = weights[n]
    elif init_type == "edge_site_left":
        n_zeros_left = 0
        n_zeros_right = n_sites-1
        states,weights = generate_states(Nb_max,[1.0])
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

##### create Hamiltonian to evolve unitarily
H_static = [
        ["+-",hop_list],
        ["+-",hop_list_hc],
        ["n",E_list],
        ["++--",int_list]
        ]

# hamiltonian
H=hamiltonian(H_static,[],basis=basis,dtype=np.complex128)

state_idxs = []
state_labels = []
state_ints = []

for state in basis.states:
    state_idxs.append(basis.index(state))
    state_labels.append(basis.int_to_state(state))
    state_ints.append(state)

state_idxs = np.asarray(state_idxs)
state_ints = np.asarray(state_ints)



##### create Lindbladian
# 1 is annihilation, 2 is creation
# site-coupling lists
L_left1_list=[[1.0,0]]
L_left2_list=[[1.0,0]]
L_right1_list=[[1.0,n_sites-1]]
L_right2_list=[[1.0,n_sites-1]]

# static opstr list 
L_left1_static=[['-',L_left1_list]]
L_left2_static=[['+',L_left2_list]]
L_right1_static=[['-',L_right1_list]]
L_right2_static=[['+',L_right2_list]]

# Lindblad operator
L_left1=hamiltonian(L_left1_static,[],basis=basis,dtype=np.complex128,check_herm=False)
L_left2=hamiltonian(L_left2_static,[],basis=basis,dtype=np.complex128,check_herm=False)
L_right1=hamiltonian(L_right1_static,[],basis=basis,dtype=np.complex128,check_herm=False)
L_right2=hamiltonian(L_right2_static,[],basis=basis,dtype=np.complex128,check_herm=False)

print(L_right1.static)

# pre-compute operators for efficiency
L_left1_dagger=L_left1.getH()
L_left2_dagger=L_left2.getH()
L_right1_dagger=L_right1.getH()
L_right2_dagger=L_right2.getH()

L_daggerL_left1=L_left1_dagger*L_left1
L_daggerL_left2=L_left2_dagger*L_left2
L_daggerL_right1=L_right1_dagger*L_right1
L_daggerL_right2=L_right2_dagger*L_right2
#
#### determine the corresponding matvec routines ####
#
matvec_H = get_matvec_function(H.static)
matvec_L=get_matvec_function(L_right1.static)
#
#
def Lindblad_EOM_v3(time,rho,rho_out,rho_aux):
    """
    This function solves the complex-valued time-dependent GPE:
    $$ \dot\rho(t) = -i[H,\rho(t)] + 2\gamma\left( L\rho L^\dagger - \frac{1}{2}\{L^\dagger L, \rho \} \right) $$
    """
    rho = rho.reshape((H.Ns,H.Ns)) # reshape vector from ODE solver input
    ### Hamiltonian part
    # commutator term (unitary
    # rho_out = H.static.dot(rho))
    matvec_H(H.static  ,rho  ,out=rho_out  ,a=+1.0,overwrite_out=True)
    # rho_out -= (H.static.T.dot(rho.T)).T // RHS~rho.dot(H) 
    matvec_H(H.static.T,rho.T,out=rho_out.T,a=-1.0,overwrite_out=False)
    # 
    for func,Hd in iteritems(H._dynamic):
            ft = func(time)
            # rho_out += ft*Hd.dot(rho)
            matvec_H(Hd  ,rho  ,out=rho_out  ,a=+ft,overwrite_out=False)
            # rho_out -= ft*(Hd.T.dot(rho.T)).T 
            matvec_H(Hd.T,rho.T,out=rho_out.T,a=-ft,overwrite_out=False)
    # multiply by -i
    rho_out *= -1.0j

    # Left lead annihilation
    #matvec(L_left1.static  ,rho             ,out=rho_aux  ,a=+2.0*gammaL1,overwrite_out=True)
    #matvec(L_left1.static.T.conj(),rho_aux.T,out=rho_out.T,a=+1.0,overwrite_out=False) 
    #matvec(L_daggerL_left1.static  ,rho  ,out=rho_out  ,a=-gammaL1,overwrite_out=False)
    #matvec(L_daggerL_left1.static.T,rho.T,out=rho_out.T,a=-gammaL1,overwrite_out=False) 

    # Left lead creation
    #matvec(L_left2.static  ,rho             ,out=rho_aux  ,a=+2.0*gammaL2,overwrite_out=True)
    #matvec(L_left2.static.T.conj(),rho_aux.T,out=rho_out.T,a=+1.0,overwrite_out=False) 
    #matvec(L_daggerL_left2.static  ,rho  ,out=rho_out  ,a=-gammaL2,overwrite_out=False)
    #matvec(L_daggerL_left2.static.T,rho.T,out=rho_out.T,a=-gammaL2,overwrite_out=False) 

    # Right lead annihilation
    matvec_L(L_right1.static  ,rho             ,out=rho_aux  ,a=+2.0*gammaR1,overwrite_out=True)
    matvec_L(L_right1.static.T.conj(),rho_aux.T,out=rho_out.T,a=+1.0,overwrite_out=False) 
    matvec_L(L_daggerL_right1.static  ,rho  ,out=rho_out  ,a=-gammaR1,overwrite_out=False)
    matvec_L(L_daggerL_right1.static.T,rho.T,out=rho_out.T,a=-gammaR1,overwrite_out=False) 

    # Right lead creation
    matvec_L(L_right2.static  ,rho             ,out=rho_aux  ,a=+2.0*gammaR2,overwrite_out=True)
    matvec_L(L_right2.static.T.conj(),rho_aux.T,out=rho_out.T,a=+1.0,overwrite_out=False) 
    matvec_L(L_daggerL_right2.static  ,rho  ,out=rho_out  ,a=-gammaR2,overwrite_out=False)
    matvec_L(L_daggerL_right2.static.T,rho.T,out=rho_out.T,a=-gammaR2,overwrite_out=False) 


    return rho_out.ravel() # ODE solver accepts vectors only
#
# define auxiliary arguments
EOM_args=(np.zeros((H.Ns,H.Ns),dtype=np.complex128,order="C"),    # auxiliary variable rho_out
		  np.zeros((H.Ns,H.Ns),dtype=np.complex128,order="C"),  ) # auxiliary variable rho_aux
#
##### time-evolve state according to Lindlad equation

print(np.linalg.norm(psi0))

rho0 = np.outer(psi0,psi0)
rho_t = evolve(rho0,ts[0],ts,Lindblad_EOM_v3,f_params=EOM_args,iterate=False,atol=1E-12,rtol=1E-12) 

print(rho_t.shape)
# setting up observables
# site occupation numbers
n_list = [hamiltonian([["n",[[1.0,i]]]],[],basis=basis,dtype=np.complex128) for i in range(n_sites)]



ns = np.zeros([n_sites,num])
for n in range(num):
    for m in range(n_sites):
        ns[m,n] = np.real(np.trace(np.matmul(rho_t[:,:,n],n_list[m].toarray())))
print(ns.shape)

Es = np.zeros([Nb_max+1,num])

for n in range(num):
    for m in range(Nb_max+1):
        indices = n_particle_indices(basis,m,n_sites)
        Es[m,n] = np.real(np.trace(np.matmul(rho_t[indices,:,n][:,indices],H.toarray()[indices,:][:,indices])))

    
output_name = "lindblad"+\
        "_lossR{:.4f}".format(gammaR1)+\
        "_"+init_type+\
        "_Nb_max"+str(Nb_max)+\
        "_U{:.4f}".format(U)+\
        "_VB{:.3f}".format(VB)


file = h5py.File('./Data/'+name+output_name+'.h5','w')
file.create_dataset('lattice_name',data=lattice_name,dtype=h5py.string_dtype(encoding='utf-8'))
file.create_dataset('init_type',data=init_type,dtype=h5py.string_dtype(encoding='utf-8'))
file.create_dataset('Nb_max',data=Nb_max)
file.create_dataset('n_sites',data=n_sites)
file.create_dataset('ts',data=ts)
file.create_dataset('U',data=U)
if lattice_name == "three_site":
    file.create_dataset('rAB',data=rAB)
if lattice_name == "diamond":
    file.create_dataset('flux',data=flux)
if lattice_name == "sawtooth":
    file.create_dataset('VB',data=VB)
if lattice_name != "three_site":
    file.create_dataset('n_cells',data=n_cells,dtype='i8')
file.create_dataset('gammaR',data=gammaR1)
file.create_dataset('rho_t_r',data=np.real(rho_t))
file.create_dataset('rho_t_i',data=np.imag(rho_t))
file.create_dataset('ns',data=np.real(ns))

file.create_dataset('state_ints',data=state_ints,dtype='i8')
file.create_dataset('state_idxs',data=state_idxs,dtype='i8')
file.create_dataset('state_labels',data=state_labels,dtype=h5py.string_dtype(encoding='utf-8'))
file.close()






