import numpy as np
from matplotlib import pyplot as plt
import h5py
import sys
import math
import glob
from quspin.basis import boson_basis_1d # bosonic Hilbert space
from scipy.optimize import curve_fit

from scipy import signal

from scipy.signal import find_peaks

import matplotlib as mpl

import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm

SMALL_SIZE = 22
MEDIUM_SIZE = 23
BIGGER_SIZE = 25
LEGEND_SIZE = 18

#SMALL_SIZE = 11
#MEDIUM_SIZE = 11
#BIGGER_SIZE = 13
#LEGEND_SIZE = 10

#figure_width = (8.6/2.56)
#figure_height = figure_width*4.8/6.4
figure_width = 5.0
figure_height = 4.8

mpl.rcParams['axes.titlesize'] = SMALL_SIZE     # fontsize of the axes title
mpl.rcParams['axes.labelsize'] = MEDIUM_SIZE    # fontsize of the x and y labels
mpl.rcParams['xtick.labelsize'] = SMALL_SIZE    # fontsize of the tick labels
mpl.rcParams['xtick.major.size'] = 4    # fontsize of the tick labels
mpl.rcParams['xtick.major.width'] = 1.0    # fontsize of the tick labels
mpl.rcParams['xtick.minor.size'] = 2.75    # fontsize of the tick labels
mpl.rcParams['xtick.minor.width'] = 0.8    # fontsize of the tick labels
mpl.rcParams['xtick.minor.visible'] = True    # fontsize of the tick labels
mpl.rcParams['ytick.labelsize'] = SMALL_SIZE    # fontsize of the tick labels
mpl.rcParams['ytick.major.size'] = 4    
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['ytick.minor.size'] = 2.75    # fontsize of the tick labels
mpl.rcParams['ytick.minor.width'] = 0.8    # fontsize of the tick labels
mpl.rcParams['ytick.minor.visible'] = True    # fontsize of the tick labels
mpl.rcParams['legend.fontsize'] = LEGEND_SIZE    # legend fontsize
mpl.rcParams['figure.titlesize'] = BIGGER_SIZE  # fontsize of the figure title
mpl.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = "Computer Modern Roman"
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['lines.linewidth'] = 2.0

mpl.rcParams['figure.figsize'] = [figure_width,figure_height]
#mpl.rcParams['font.family'] = "Helvetica"
#mpl.rcParams.update({'pgf.preamble':r'\usepackage{amsmath}'})
#mpl.rcParams['text.latex.preamble'] = [
#r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
#r'\usepackage{amsmath}',
#       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
#       r'\usepackage{helvet}',    # set the normal font here
#       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
#       r'\sansmath'              
#]

import distinctipy

import cmasher as cmr
import colorcet as cc

#bcolors =[(166.0/256,206.0/256,227.0/256),\
#        (31.0/256,120.0/256,180.0/256),\
#        (178.0/256,223.0/256,138.0/256),\
#        (51.0/256,160.0/256,44.0/256)]

bcolors = [(27.0/256,158.0/256,119.0/256),\
        (217.0/256,95.0/256,2.0/256),\
        (117.0/256,112.0/256,179.0/256),\
        (231.0/256,41.0/256,138.0/256)]

bcolors2 = [(228.0/256,26.0/256,28.0/256),\
    (55.0/256,126.0/256,184.0/256),\
    (77.0/256,175.0/256,74.0/256),\
    (152.0/256,78.0/256,163.0/256)]




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


def n_particle_indices(basis,n,n_sites):
    indices = []
    states, weights = generate_states(n,np.ones([n_sites],dtype=complex))
    for state in states:
        indices.append(basis.index(state))
    return indices

def find_nearest(array,value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]

def probability_pnum_sites(sites,pnum,psi_t,state_labels):
    states = []
    pnums_at_sites_states = [list(map(int,list(label[1:-1].replace(' ','')))) for label in state_labels]

    for count,pnums_at_sites in enumerate(pnums_at_sites_states):
        print(pnums_at_sites)
        pnums_at_sites = np.asarray(pnums_at_sites)
        pnum_at_sites = np.sum(pnums_at_sites[sites])
        print(pnum_at_sites)
        if pnum_at_sites == pnum:
            states.append(count)
    print("states",states)
    output = np.zeros(psi_t.shape[1])
    for state in states:
        output += np.abs(psi_t[state,:])**2
    return output





path = sys.argv[1]


n_sites = 3
Nb = 2

f = h5py.File(path,'r')
lattice_name = f['lattice_name'].asstr()[...]
print(lattice_name)
init_type = f['init_type'].asstr()[...]
print(init_type)
Nb = int(f['Nb'][()])
ts = f['ts'][:]
n_sites = int(f['n_sites'][()])

print("n_sites",n_sites)


basis = boson_basis_1d(n_sites,Nb=Nb) 

start = ts[0]
stop = ts[-1]
steps = ts.shape[0]

overlaps = f['overlaps_r'][:]+1.0j*f['overlaps_i'][:]
eigenvalues = f['eigenvalues'][:]
eigenvectors = f['eigenvectors_r'][:]+1.0j*f['eigenvectors_i'][:]

U = f['U'][()]
rAB = 0
if lattice_name == 'three_site':
    rAB = f['rAB'][()]
flux = 0
n_cells = 1
if lattice_name == 'diamond':
    flux = f['flux'][()]
    n_cells = f['n_cells'][()]
if lattice_name == 'sawtooth':
    n_cells = f['n_cells'][()]

psi_t = f['psi_t_r'][:]+1.0j*f['psi_t_i'][:]
ns_t = f['ns_t'][:]


state_ints = f['state_ints'][:]
state_idxs = f['state_idxs'][:]
state_labels = f['state_labels'].asstr()[...]
print(state_labels)
f.close()

#decay_rates.append(-np.polyfit(ts[1000:],np.log(rho_2p[1000:]),1)[0])
#ns_output_t = np.sum(ns_t[-1,:],0)
ns_output_t = ns_t[-1,:]
print(ns_t[0,:])
print(ns_t[1,:])
print(ns_t[2,:])
ns_output_fft = np.fft.fft(ns_output_t)

peaks,_ = find_peaks(np.abs(ns_output_fft))
duration = ts[-1]
peak_heights = np.abs(ns_output_fft[peaks])
max_peak_loc = np.argmax(peak_heights[0:int(peak_heights.shape[0]/2)])
#omega = peaks[0]*2*np.pi/duration
omega = max_peak_loc*2*np.pi/duration

fft_mod = np.zeros(ns_output_fft.size,dtype=complex)
fft_mod[0] = ns_output_fft[0]
fft_mod[peaks[0]-10:peaks[0]+10] = ns_output_fft[peaks[0]-10:peaks[0]+10]
fft_mod[-peaks[0]-10:-peaks[0]+10] = ns_output_fft[-peaks[0]-10:-peaks[0]+10]

ns_output_mod = np.fft.ifft(fft_mod)

#butter_low = signal.butter(4,2*omega/(2*np.pi),'lp',output='sos',fs=ns_output_fft.size/duration)
#ns_output_filtered = signal.sosfilt(butter_low,ns_output_t)


fig,ax = plt.subplots()
ax.plot(ts,np.abs(ns_output_t))


fig2,ax2 = plt.subplots()
ax2.plot(np.real(ns_output_fft))
ax2.plot(np.imag(ns_output_fft))

#ax.plot(ts,np.abs(ns_output_filtered))


#omega = peaks[0]*2*np.pi/duration
#T = 2*np.pi/omega
#print(omega)
#print(T)



fig0 = plt.figure()

overlaps_log = np.log10(np.abs(overlaps))
minimum_log = -4.0*np.ones(overlaps.shape)

#colors = np.maximum(overlaps_log,minimum_log)
colors = np.abs(overlaps)
print(np.abs(overlaps))
#plt.scatter(range(len(E)),E,c=colors,cmap=cc.cm.fire,norm=Normalize(vmin=1e-2,vmax=1.0,clip=True))
plt.scatter(range(len(eigenvalues)),eigenvalues,c=colors,cmap=cc.cm.fire,norm=LogNorm(vmin=1e-3,vmax=1.0,clip=True))
cbar =plt.colorbar()
cbar.set_label("Overlap with initial state")
plt.xlabel("Eigenstate index")
plt.ylabel("Energy")
#plt.title("Overlap with initial state")
#plt.savefig("./Figures/sawtooth_init_overlap_quantum_"+str(n_cells)+"cells_Nb"+str(Nb)+"_U"+str(U)+"_VB"+str(VB)+"_tAA"+str(t_AA)+"_left_edge.png",format='png')
#ax.plot(E)
#ax.plot(np.abs(overlaps))


#print("diagonal ensemble n_Aedge",n_Aedge)

# Analysis for Sawtooth lattice
fig1,ax1 = plt.subplots()
#pos = ax.pcolor(ts,np.linspace(1,n_sites+1,n_sites+1),ns,norm=colors.LogNorm(vmin=ns.min(), vmax=ns.max()),cmap='hot',shading='auto')
pos = ax1.pcolor(ts,np.linspace(1,n_sites,n_sites),ns_t,cmap=cc.cm.fire,shading='auto',norm=Normalize(vmin=1e-4,vmax=Nb,clip=True),rasterized=True)
ax1.set_xlabel("Time")
ax1.set_ylabel("Site")
ax1.set_yticks(np.arange(1,n_sites+1,step=1))
cbar = fig1.colorbar(pos,ax=ax1,pad=0.00)
cbar.set_label("Photon number")

#fig2,ax2 = plt.subplots()
#ax2.plot(ts,np.sum(np.abs(evo)**2,0))

dcolors = distinctipy.get_colors(3)
#bcolors = [(27.0/256,158.0/256,119.0/256),(217.0/256,95.0/256,2.0/256),(117.0/256,112.0/256,179.0/256)]
bcolors = [(27.0/256,158.0/256,119.0/256),\
        (217.0/256,95.0/256,2.0/256),\
        (117.0/256,112.0/256,179.0/256),\
        (231.0/256,41.0/256,138.0/256)]

print("n_cells",n_cells)

fig3,ax3 = plt.subplots()
if lattice_name == "diamond" and n_cells > 1:
    ax3.plot(ts,np.sum(ns_t[0:3,:],0), color = bcolors[0], label="1+2+3")
    ax3.plot(ts,np.sum(ns_t[-3:,:],0), color = bcolors[1], label=str(n_sites-2)+"+"+str(n_sites-1)+"+"+str(n_sites))
elif lattice_name == "diamond" and n_cells == 1:
    ax3.plot(ts,ns_t[0,:],color = bcolors[0], label ="Left edge site")
    ax3.plot(ts,np.sum(ns_t[1:-1,:],0),color = bcolors[1], label ="Middle sites")
    ax3.plot(ts,ns_t[-1,:],color = bcolors[2], label ="Right edge site")
elif lattice_name == "sawtooth" and n_cells > 1:
    ax3.plot(ts,np.sum(ns_t[0:2,:],0), color = bcolors[0], label="1+2")
    ax3.plot(ts,np.sum(ns_t[-2:,:],0), color = bcolors[1], label=str(n_sites-1)+"+"+str(n_sites))
elif lattice_name == "sawtooth" and n_cells == 1:
    ax3.plot(ts,ns_t[0,:],color = bcolors[0], label ="Left edge site")
    ax3.plot(ts,ns_t[1,:],color = bcolors[1], label ="Middle site")
    ax3.plot(ts,ns_t[-1,:],color = bcolors[2], label ="Right edge site")
    #ax3.plot(ts,ns_t[-1,:]+ns_t[1,:],color = bcolors[2], label ="B+C")
elif lattice_name == "three_site":
    ax3.plot(ts,ns_t[0,:],color = bcolors[0], label ="Site 1")
    ax3.plot(ts,np.sum(ns_t[1:,:],0),color = bcolors[1], label ="Sites 2+3")
    #ax3.plot(ts,ns_t[-1,:],color = bcolors[2], label ="Site 3")

#ax3.plot(ts,np.sum(ns[0:2,:],0)/Nb)
ax3.set_xlim([start,stop])
ax3.set_ylim([1e-6,Nb+0.05])
ax3.set_xlabel("Time")
ax3.set_ylabel("Photon number")
ax3.legend()


#fig4, ax4 = plt.subplots()
#for state in basis.states:
#    ax4.plot(ts,np.abs(psi_t[basis.index(state),:])**2,label=basis.int_to_state(state))
#ax4.legend()

if Nb == 2:
    if lattice_name == "three_site":
        BC_zero = probability_pnum_sites([1,2],0,psi_t,state_labels)
        BC_single = probability_pnum_sites([1,2],1,psi_t,state_labels)
        BC_double = probability_pnum_sites([1,2],2,psi_t,state_labels)
        #BC_single = np.abs(psi_t[1,:])**2+np.abs(psi_t[2,:])**2
        #BC_double = np.abs(psi_t[3,:])**2+np.abs(psi_t[4,:])**2+np.abs(psi_t[5,:])**2
        fig5, ax5 = plt.subplots()
        fig5.set_figwidth(figure_width)
        fig5.set_figheight(2.8)
        #ax5.plot(ts,np.abs(psi_t[0,:])**2,label=r'$P_0^{\mathrm{B\!+\!C}}$',color=bcolors[0],zorder=2)
        ax5.plot(ts,BC_zero,label=r'$P_0^{\mathrm{B\!+\!C}}$',color=bcolors[0],zorder=2)
        ax5.plot(ts,BC_single+BC_double,label=r'$P_{1+2}^{\mathrm{B\!+\!C}}$',color = bcolors[3],zorder=0)
        #ax5.plot(ts,BC_double,label=r'$P_2^{\mathrm{B\!+\!C}}$',color = bcolors[3],zorder=1)
        ax5.set_xlabel(r"Time ($1/t_{AC}$)")
        ax5.set_ylabel(r"Probability")
        ax5.legend(loc='upper right')
        ax5.set_xlim(ts[0],ts[-1])
        ax5.set_ylim(0,1)
        
        fig6, ax6 = plt.subplots()
        ax6.plot(ts,BC_double+BC_single)

    if lattice_name == "diamond":
        print(n_cells)
        if n_cells >= 2:
            right_zero = probability_pnum_sites([-3,-2,-1],0,psi_t,state_labels)
            right_single = probability_pnum_sites([-3,-2,-1],1,psi_t,state_labels)
            right_double = probability_pnum_sites([-3,-2,-1],2,psi_t,state_labels)
        if n_cells == 1:
            right_zero = probability_pnum_sites([-1],0,psi_t,state_labels)
            right_single = probability_pnum_sites([-1],1,psi_t,state_labels)
            right_double = probability_pnum_sites([-1],2,psi_t,state_labels)
        fig5, ax5 = plt.subplots()

        fig5.set_figwidth(figure_width)
        fig5.set_figheight(2.8)

        ax5.plot(ts,right_zero,label=r'$P_0^{\mathrm{right~edge}}$',color=bcolors[0],zorder=2)
        ax5.plot(ts,right_single,label=r'$P_{1}^{\mathrm{right~edge}}$',color = bcolors[2],zorder=0)
        ax5.plot(ts,right_double,label=r'$P_2^{\mathrm{right~edge}}$',color = bcolors[3],zorder=1)
        ax5.set_xlabel(r"Time ($1/J$)")
        ax5.set_ylabel(r"Probability")
        ax5.legend(loc='upper right')
        ax5.set_xlim(ts[0],ts[-1])
        ax5.set_ylim(0,1)
        
        fig6, ax6 = plt.subplots()
        ax6.plot(ts,right_double+right_single)

if Nb == 1:
    if lattice_name == "three_site":
        BC_prob = np.abs(psi_t[1,:])**2 + np.abs(psi_t[2,:])**2
        A_prob = np.abs(psi_t[0,:])**2
        fig5, ax5 = plt.subplots()
        fig5.set_figwidth(figure_width)
        fig5.set_figheight(2.8)
        ax5.plot(ts,BC_prob,label='$P_{1}^{\mathrm{B\!+\!C}}$',color=bcolors[0])
        ax5.plot(ts,A_prob,label='$P_{0}^{\mathrm{B\!+\!C}}$',color=bcolors[3])
        ax5.legend(loc='upper right')
        ax5.set_xlim(ts[0],ts[-1])
        ax5.set_ylim(0,1)
        ax5.set_xlabel("Time ($1/t_{AC}$)")
        ax5.set_ylabel("Probability")

    if lattice_name == "diamond":
        if n_cells >= 2:
            left_prob = np.abs(psi_t[0,:])**2 + np.abs(psi_t[1,:])**2+np.abs(psi_t[2,:])**2
            right_prob = np.abs(psi_t[-3,:])**2 + np.abs(psi_t[-2,:])**2+np.abs(psi_t[-1,:])**2
        else:
            left_prob = np.abs(psi_t[0,:])**2 + np.abs(psi_t[1,:])**2+np.abs(psi_t[2,:])**2
            right_prob = np.abs(psi_t[-1,:])**2 
        fig5, ax5 = plt.subplots()
        fig5.set_figwidth(figure_width)
        fig5.set_figheight(2.8)
        ax5.plot(ts,left_prob,label='$P_{1}^{\mathrm{right~edge}}$',color=bcolors[0])
        ax5.plot(ts,right_prob,label='$P_{0}^{\mathrm{right~edge}}$',color=bcolors[3])
        ax5.legend(loc='upper right')
        ax5.set_xlim(ts[0],ts[-1])
        ax5.set_ylim(0,1)
        ax5.set_xlabel("Time ($1/t$)")
        ax5.set_ylabel("Probability")

if len(sys.argv) == 3:
    n_sites2 = 3
    Nb2 = 2
    path2 = sys.argv[2]
    
    f2 = h5py.File(path2,'r')
    lattice_name2 = f2['lattice_name'].asstr()[...]
    init_type2 = f2['init_type'].asstr()[...]
    print(init_type)
    Nb2 = int(f2['Nb'][()])
    ts2 = f2['ts'][:]
    n_sites2 = int(f2['n_sites'][()])

    
    
    basis2 = boson_basis_1d(n_sites2,Nb=Nb2) 
    
    start2 = ts2[0]
    stop2 = ts2[-1]
    steps2 = ts2.shape[0]
    
    overlaps2 = f2['overlaps_r'][:]+1.0j*f2['overlaps_i'][:]
    eigenvalues2 = f2['eigenvalues'][:]
    eigenvectors2 = f2['eigenvectors_r'][:]+1.0j*f2['eigenvectors_i'][:]
    
    U2 = f2['U'][()]
    rAB2 = 0
    if lattice_name2 == 'three_site':
        rAB2 = f2['rAB'][()]
    flux2 = 0
    if lattice_name2 == 'diamond':
        flux2 = f2['flux'][()]
    n_cells2 = 1
    if lattice_name2 != 'three_site':
        n_cells2 = f2['n_cells'][()]
    psi_t2 = f2['psi_t_r'][:]+1.0j*f2['psi_t_i'][:]
    ns_t2 = f2['ns_t'][:]
    
    
    state_ints2 = f2['state_ints'][:]
    state_idxs2 = f2['state_idxs'][:]
    state_labels2 = f2['state_labels'].asstr()[...]
    f2.close()
    fig1,(ax1,ax2) = plt.subplots(nrows=2,sharex=True)
    #pos = ax.pcolor(ts,np.linspace(1,n_sites+1,n_sites+1),ns,norm=colors.LogNorm(vmin=ns.min(), vmax=ns.max()),cmap='hot',shading='auto')
    pos = ax1.pcolor(ts,np.linspace(1,n_sites,n_sites),ns_t/Nb,cmap=cc.cm.fire,shading='auto',norm=Normalize(vmin=1e-4,vmax=1,clip=True),rasterized=True)
    pos2 = ax2.pcolor(ts,np.linspace(1,n_sites2,n_sites2),ns_t2/Nb2,cmap=cc.cm.fire,shading='auto',norm=Normalize(vmin=1e-4,vmax=1,clip=True),rasterized=True)
    fig1.supylabel("Site")
    ax1.set_yticks(np.arange(1,n_sites+1,step=1))
    ax2.set_yticks(np.arange(1,n_sites2+1,step=1))
    if lattice_name == "three_site":
        ax1.set_yticklabels(['A','B','C'])
        ax2.set_yticklabels(['A','B','C'])
        ax2.set_xlabel("Time (1/$t_{AC}$)")
    if lattice_name == "diamond":
        ax2.set_xlabel("Time (1/$J$)")
    cbar = fig1.colorbar(pos,ax=[ax1,ax2],pad=0.00)

    if Nb2 == 2 and lattice_name=="three_site":
        BC_single2 = np.abs(psi_t2[1,:])**2+np.abs(psi_t2[2,:])**2
        BC_double2 = np.abs(psi_t2[3,:])**2+np.abs(psi_t2[4,:])**2+np.abs(psi_t2[5,:])**2
        
    
    if Nb2 == 1:
        if lattice_name == "three_site":
            BC_prob2 = np.abs(psi_t2[1,:])**2 + np.abs(psi_t2[2,:])**2
            A_prob = np.abs(psi_t[0,:])**2
        if lattice_name == "diamond":
            if n_cells >= 2:
                right_zero2 = probability_pnum_sites([-3,-2,-1],0,psi_t2,state_labels2)
                right_single2 = probability_pnum_sites([-3,-2,-1],1,psi_t2,state_labels2)
            else:
                right_zero2 = probability_pnum_sites([-1],0,psi_t2,state_labels2)
                right_single2 = probability_pnum_sites([-1],1,psi_t2,state_labels2)

    fig5, (ax1,ax2) = plt.subplots(nrows=2,sharex=True)
    #fig5.set_figwidth(6.4)
    #fig5.set_figheight(2.4)
    if lattice_name == "three_site":
        ax1.plot(ts,np.abs(psi_t[0,:])**2,label=r'$P_0^{\mathrm{B\!+\!C}}$',color=bcolors[0],zorder=2)
        ax1.plot(ts,BC_single+BC_double,label=r'$P_{1+2}^{\mathrm{B\!+\!C}}$',color = bcolors[3],zorder=0)
    if lattice_name == "diamond":
        ax1.plot(ts,right_zero,label=r'$P_0^{\mathrm{right~edge}}$',color=bcolors[0],zorder=2)
        ax1.plot(ts,right_single,label=r'$P_{1}^{\mathrm{right~edge}}$',color = bcolors[2],zorder=0)
        ax1.plot(ts,right_double,label=r'$P_{2}^{\mathrm{right~edge}}$',color = bcolors[3],zorder=0)

    #ax5.plot(ts,BC_double,label=r'$P_2^{\mathrm{B\!+\!C}}$',color = bcolors[3],zorder=1)
    ax1.legend(loc='upper right')
    ax1.set_xlim(ts[0],ts[-1])
    ax1.set_ylim(0,1)

    if lattice_name == "three_site":
        ax2.plot(ts,np.abs(psi_t2[0,:])**2,label=r'$P_0^{\mathrm{B\!+\!C}}$',color=bcolors[0],zorder=2)
        ax2.plot(ts,BC_prob2,label=r'$P_{1}^{\mathrm{B\!+\!C}}$',color = bcolors[3],zorder=0)
    if lattice_name == "diamond":
        ax2.plot(ts,right_zero2,label=r'$P_0^{\mathrm{right~edge}}$',color=bcolors[0],zorder=2)
        ax2.plot(ts,right_single2,label=r'$P_{1}^{\mathrm{right~edge}}$',color = bcolors[3],zorder=0)
    if lattice_name == "three_site":
        ax2.set_xlabel(r"Time ($1/t_{AC}$)")
    if lattice_name == "diamond":
        ax2.set_xlabel(r"Time ($1/J$)")

    ax2.legend(loc="upper right")
    fig5.supylabel("Probability")

plt.show()






#fig0.savefig("./Figures/"+name+"_overlaps_Nb"+str(Nb)+"_U"+str(U)+"_VB"+str(VB)+"_"+init_type+".svg",format='svg')
#fig1.savefig("./Figures/"+name+"_pnum_vs_site_time_Nb"+str(Nb)+"_U"+str(U)+"_VB"+str(VB)+"_"+init_type+".svg",format='svg')
#fig3.savefig("./Figures/"+name+"_pnum_vs_time_Nb"+str(Nb)+"_U"+str(U)+"_VB"+str(VB)+"_"+init_type+".svg",format='svg')


plt.show()

