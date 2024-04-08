# Code for numerical integration of the non-linear Bose-Hubbard equations
# of motion for the boson field in the \hat{b} -> b approximation.
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib import colors

from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from scipy import integrate as inte

import matplotlib as mpl

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


# hopping matrix for the sawtooth lattice
def hop_sawtooth(n_cells,tAA,tAB):
    n_sites = 2*n_cells + 1
    hop = np.zeros([n_sites,n_sites],dtype=complex)
    for n in range(n_cells):
        hop[2*(n+1),2*n] = tAA
        hop[2*n,2*(n+1)] = np.conj(tAA)
        hop[2*n+1,2*n] = tAB
        hop[2*n,2*n+1] = np.conj(tAB)
        hop[2*(n+1),2*n+1] = tAB
        hop[2*n+1,2*(n+1)] = np.conj(tAB)
    return hop

def hop_linear(n_cells,t):
    n_sites = n_cells
    hop = np.zeros([n_sites,n_sites],dtype=complex)
    for n in range(n_cells-1):
        hop[n+1,n] = t
        hop[n,n+1] = np.conj(t)
    return hop

def hop_diamond(n_cells,flux):
    n_sites = 3*n_cells+1
    hop = np.zeros([n_sites,n_sites],dtype=complex)
    for n in range(n_cells):
        # Convention of PHYSICAL REVIEW A 99, 013826 (2019)
        # hop[3*n,3*n+1] = np.exp(-1.0j*flux/2)  # a_n,b_n
        # hop[3*n+1,3*n] = np.exp(1.0j*flux/2)  # b_n,a_n
        # hop[3*n,3*n+2] = 1.0 # a_n,c_n
        # hop[3*n+2,3*n] = 1.0 # c_n,a_n
        # hop[3*n+1,3*(n+1)] = 1.0 # b_n,a_{n+1}
        # hop[3*(n+1),3*n+1] = 1.0 # a_{n+1},b_n
        # hop[3*n+2,3*(n+1)] = np.exp(1.0j*flux/2) # c_n,a_{n+1}
        # hop[3*(n+1),3*n+2] = np.exp(-1.0j*flux/2) # a_{n+1},c_n
        # Convention of PHYSICAL REVIEW A 100, 043829 (2019)
        # hop[3*n,3*n+1] = np.exp(1.0j*flux)  # a_n,b_n
        # hop[3*n+1,3*n] = np.exp(-1.0j*flux)  # b_n,a_n
        # hop[3*n,3*n+2] = 1.0 # a_n,c_n
        # hop[3*n+2,3*n] = 1.0 # c_n,a_n
        # hop[3*n+1,3*(n+1)] = 1.0 # b_n,a_{n+1}
        # hop[3*(n+1),3*n+1] = 1.0 # a_{n+1},b_n
        # hop[3*n+2,3*(n+1)] = 1.0 # c_n,a_{n+1}
        # hop[3*(n+1),3*n+2] = 1.0 # a_{n+1},c_n
        # Own convention :)
        hop[3*n,3*n+1] = -1.0 # a_n,b_n
        hop[3*n+1,3*n] = -1.0 # b_n,a_n
        hop[3*n,3*n+2] = -1.0 # a_n,c_n
        hop[3*n+2,3*n] = -1.0 # c_n,a_n
        hop[3*n+1,3*(n+1)] = -np.exp(1.0j*flux) # b_n,a_{n+1}
        hop[3*(n+1),3*n+1] = -np.exp(-1.0j*flux) # a_{n+1},b_n
        hop[3*n+2,3*(n+1)] = -1.0 # c_n,a_{n+1}
        hop[3*(n+1),3*n+2] = -1.0 # a_{n+1},c_n
    return hop

def hop_three_site(rAB,tAC,tAB,epsC):
    n_sites = 3
    tBC = -rAB*tAC
    epsA = tAB*(tBC/tAC-tAC/tBC)
    hop = np.zeros([n_sites,n_sites],dtype=complex)
    hop[0,2] = -tAC
    hop[2,0] = -np.conj(tAC)
    hop[0,1] = -tAB
    hop[1,0] = -np.conj(tAB)
    hop[1,2] = -tBC
    hop[2,1] = -np.conj(tBC)
    hop[0,0] = epsA
    hop[2,2] = epsC

    return hop



# Calculate the non-linear Hamiltonian with given field values b
def Hb(b,hop,U):
    n_sites = b.size
    nl_coeff = 2.0*U*np.abs(b)**2
    Hb = np.matmul(hop,b) + nl_coeff*b
    return Hb

# model independent parameters
Np = 1 # number of particles
n_cells = 2
U = 1.0
gain = 0.0
loss = 0.1

logscale = False

# Start and stop time
start = 0.0
stop = 1000.0
max_step = (stop-start)/1000

lattice = "three_site"
init_type = "left_edge_site"

if lattice == "three_site":
    # model parameters, sawtooth
    rAB = -5
    name = "three_site_"+str(rAB)+"_"
    n_sites = 3
    tAC = -1.0
    tAB = 0.0
    VB = 0.0
    epsC = 0.0
    hop = hop_three_site(rAB,tAC,tAB,epsC)
    hop[-1,-1] += -1.0j*loss
    hop[0,0] += 1.0j*gain
    
    b0 = np.zeros(n_sites,dtype=complex)
    if init_type == "left_edge_site":
        b0[0] = 1.0
    if init_type == "left_edge_state":
        b0[0] = rAB
        b0[1] = 1.0
    b0 *= np.sqrt(Np)/np.linalg.norm(b0)

if lattice == "sawtooth":
    # model parameters, sawtooth
    name = "sawtooth"+str(n_cells)
    n_sites = 2*n_cells+1
    tAA = -1.0
    tAB = -np.sqrt(2)
    VB = -1.0*tAA
    mu = 0.0
    hop = hop_sawtooth(n_cells,tAA,tAB)
    np.fill_diagonal(hop,-mu)
    hop[0,0] += VB
    hop[-1,-1] += VB
    hop[-1,-1] += -1.0j*loss
    hop[0,0] += 1.0j*gain
    init_loc = 2
    
    eigs,vecs = np.linalg.eigh(-hop)
    print(eigs)
    
    b0 = np.zeros(n_sites,dtype=complex)
    if init_type == "left_edge_site":
        b0[0] = np.sqrt(Np)
    if init_type == "left_edge_state":
        b0[0] = np.sqrt(2.0/3.0)*np.sqrt(Np)
        b0[1] = -np.sqrt(1.0/3.0)*np.sqrt(Np)
    if init_type == "Vstate":
        b0[init_loc*2] = -np.sqrt(2)
        b0[init_loc*2+1] = 1.0
        b0[init_loc*2-1] = 1.0
        b0 *= np.sqrt(Np)/np.linalg.norm(b0)

if lattice == "diamond":
    # model parameters, diamond
    name = "diamond"+str(n_cells)
    n_sites = 3*n_cells+1
    flux = np.pi
    mu = 0.0
    #alpha = 1.0/0.1
    alpha = 1/0.1
    init_loc = 1
    hop = hop_diamond(n_cells,flux)
    np.fill_diagonal(hop,-mu)
    VB = 0.0

    hop[-1,-1] += -1.0j*loss
    hop[0,0] += 1.0j*gain
    
    b0 = np.zeros(n_sites,dtype=complex)
    if init_type == "ABcage":
        b0[init_loc*3] = alpha
        b0[init_loc*3+1] = -1.0
        b0[init_loc*3+2] =  1.0 
        b0[init_loc*3-1] =  1.0 
        b0[init_loc*3-2] =  1.0 
        b0 *= np.sqrt(Np)/np.linalg.norm(b0)
    if init_type == "left_edge_site":
        b0[0] = np.sqrt(Np)
    if init_type == "left_edge_state":
        b0[0] = np.sqrt(2)
        b0[1] = 1.0
        b0[2] = 1.0
        b0 *= np.sqrt(Np)/np.linalg.norm(b0)

if lattice == "linear":
    # model parameters, linear
    n_sites = n_cells
    t = -1.0
    mu = 0.0
    VB = 0.0
    hop = hop_linear(n_cells,t)
    np.fill_diagonal(hop,-mu)
    hop[-1,-1] += -1.0j*loss
    hop[0,0] += 1.0j*gain
    
    b0 = np.zeros(n_sites,dtype=complex)
    if init_type == "left_edge_site":
        b0[0] = 1.0

schr_right = lambda t,b: -1.0j*Hb(b,hop,U)


# time integration
solver = inte.RK45(schr_right,start,b0,stop,max_step=max_step,rtol=1.0e-8)
ts = [0.0]
bs = [b0]
while(solver.status != 'finished'):
    solver.step()
    ts.append(solver.t)
    bs.append(solver.y)

ts = np.asarray(ts)
bs = np.asarray(bs)

# Analysis
fig,ax = plt.subplots()
ns = np.abs(bs.transpose())**2/Np 
if logscale:
    pos = ax.pcolor(ts,np.linspace(1,n_sites,n_sites),ns,norm=colors.LogNorm(vmin=1e-6, vmax=1.0),cmap=cc.cm.fire,rasterized=True,shading='auto')
else:
    pos = ax.pcolor(ts,np.linspace(1,n_sites,n_sites),ns,cmap=cc.cm.fire,shading='auto',norm=Normalize(vmin=1e-4,vmax=Np,clip=True),rasterized=True)
ax.set_xlabel("Time")
ax.set_ylabel("Site")
ax.set_yticks(np.arange(1,n_sites+1,step=1))
cbar = fig.colorbar(pos,ax=ax,pad=0.05)
cbar.set_label("Intensity")

fig2,ax2 = plt.subplots()
ax2.plot(ts,np.sum(np.abs(bs)**2/Np,1))

dcolors = distinctipy.get_colors(3)
bcolors = [(27.0/256,158.0/256,119.0/256),(217.0/256,95.0/256,2.0/256),(117.0/256,112.0/256,179.0/256)]


bcolors = [(27.0/256,158.0/256,119.0/256),\
        (217.0/256,95.0/256,2.0/256),\
        (117.0/256,112.0/256,179.0/256),\
        (231.0/256,41.0/256,138.0/256)]

fig3,ax3 = plt.subplots()
if lattice == "diamond" and n_cells > 1:
    ax3.plot(ts,np.sum(ns[0:3,:],0), color = bcolors[0], label="1+2+3")
    ax3.plot(ts,np.sum(ns[-3:,:],0), color = bcolors[1], label=str(n_sites-2)+"+"+str(n_sites-1)+"+"+str(n_sites))
elif lattice == "diamond" and n_cells == 1:
    ax3.plot(ts,ns[0,:],color = bcolors[0], label ="Left edge site")
    ax3.plot(ts,np.sum(ns[1:-1,:],0),color = bcolors[1], label ="Middle sites")
    ax3.plot(ts,ns[-1,:],color = bcolors[2], label ="Right edge site")
elif lattice == "sawtooth" and n_cells > 1:
    ax3.plot(ts,np.sum(ns[0:2,:],0), color = bcolors[0], label="1+2")
    ax3.plot(ts,np.sum(ns[-2:,:],0), color = bcolors[1], label=str(n_sites-1)+"+"+str(n_sites))
elif lattice == "sawtooth" and n_cells == 1:
    ax3.plot(ts,ns[0,:],color = bcolors[0], label ="Left edge site")
    ax3.plot(ts,ns[1,:],color = bcolors[1], label ="Middle site")
    ax3.plot(ts,ns[-1,:],color = bcolors[2], label ="Right edge site")
elif lattice == "three_site":
    ax3.plot(ts,ns[0,:],color = bcolors[0], label ="Site 1")
    ax3.plot(ts,np.sum(ns[1:,:],0),color = bcolors[1], label ="Sites 2+3")
    #ax3.plot(ts,ns[-1,:],color = bcolors[2], label ="Site 3")

#ax3.plot(ts,np.sum(ns[0:2,:],0)/Nb)
ax3.set_xlim([start,stop])
ax3.set_ylim([1e-6,Np+0.05])
ax3.set_xlabel("Time")
ax3.set_ylabel("Intensity")
ax3.legend()

fig.savefig("./Figures/"+name+"_pnum_vs_site_time_classical_U"+str(U)+"_gammaR"+str(loss)+"_VB"+str(VB)+"_"+init_type+".svg",format='svg')
fig3.savefig("./Figures/"+name+"_pnum_vs_time_classical_U"+str(U)+"_VB"+str(VB)+"_"+init_type+".svg",format='svg')
plt.show()
