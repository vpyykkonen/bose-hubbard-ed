
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import h5py
import sys
import math
import glob
from quspin.basis import boson_basis_1d # bosonic Hilbert space
from scipy.optimize import curve_fit

from scipy import signal

from scipy.signal import find_peaks

#SMALL_SIZE = 22
SMALL_SIZE = 16
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




figure_type = sys.argv[1]
path = sys.argv[2]

files = glob.glob(path+"/*.h5")
rABs  = []
Us = []
s_times = []
contrasts = []
signal_2p_max = []
signal_1p_avg = []

f = h5py.File(files[0],'r')
n_sites = int(f["n_sites"][()])
Nb = int(f["Nb"][()])
lattice_name = f["lattice_name"].asstr()[...]
state_labels = f['state_labels'].asstr()[...]
if lattice_name == "diamond" or lattice_name == "sawtooth":
    flux = f["flux"][()]
    n_cells = int(f["n_cells"][()])
f.close()
#n_sites = 3
#Nb = 2
#lattice_name = "three_site"


output_sites = []
if lattice_name == "three_site":
    output_sites = [1,2]
if lattice_name == "diamond":
    output_sites = [-3,-2-1]
if lattice_name == "sawtooth":
    output_sites = [-2,-1]


for file in files:
    f = h5py.File(file,'r')
    if lattice_name == "three_site":
        rABs.append(f['rAB'][()])
    Us.append(f['U'][()])
    ts = f['ts'][:]
    ns_t = f['ns_t'][:]
    #ns_t = f['ns'][:]
    psi_t = f['psi_t_r'][:] + 1.0j*f['psi_t_i'][:]
    #psi_t = f['psi_r'][:] + 1.0j*f['psi_i'][:]



    prob_1p_output = probability_pnum_sites(output_sites,1,psi_t,state_labels)
    prob_2p_output = probability_pnum_sites(output_sites,2,psi_t,state_labels)

    prob_output = prob_1p_output + prob_2p_output


    #decay_rates.append(-np.polyfit(ts[1000:],np.log(rho_2p[1000:]),1)[0])
    ns_output_t = np.sum(ns_t[1:,:],0)
    if lattice_name == "three_site":
        ns_output_t = np.sum(ns_t[1:,:],0)
    if lattice_name == "diamond":
        ns_output_t = np.sum(ns_t[-3:,:],0)
    if lattice_name == "sawtooth":
        ns_output_t = np.sum(ns_t[-2:,:],0)


    ns_output_fft = np.fft.fft(ns_output_t)

    prob_output_fft = np.fft.fft(prob_output)



    #peaks,_ = find_peaks(np.abs(ns_output_fft))
    peaks,_ = find_peaks(np.abs(prob_output_fft))
    duration = ts[-1]
    #peak_heights = np.abs(ns_output_fft[peaks])
    peak_heights = np.abs(prob_output_fft[peaks])
    max_peak_loc = np.argmax(peak_heights[0:int(peak_heights.shape[0]/2)])

    #omega = peaks[0]*2*np.pi/duration
    omega = peaks[max_peak_loc]*2*np.pi/duration
    T = 2*np.pi/omega

    # Optional filtering for noisy data to extract averaged maximum
    #butter_low = signal.butter(4,2*omega/(2*np.pi),'lp',output='sos',fs=ns_output_fft.size/duration)
    #ns_output_filtered = signal.sosfilt(butter_low,ns_output_t)

    #signal_2p_max.append(np.amax(ns_output_t))
    signal_2p_max.append(np.amax(prob_output))
    
    s_times.append(T/2)
    f.close()

# another file if contrast is wanted
if figure_type == "rAB_ON_OFF_contrast":
    rABs2 = []
    if len(sys.argv) > 3:
        path2 = sys.argv[3]
    else: 
        print("Path to U=0 not give. Exiting.")
        exit()

    files2 = glob.glob(path2+"/*.h5")
    f = h5py.File(files2[0],'r')
    state_labels2 = f['state_labels'].asstr()[...]
    f.close()
    
    for file in files2:
        f = h5py.File(file,'r')
        rABs2.append(f['rAB'][()])
        ts2 = f['ts'][:]
        ns_t2 = f['ns_t'][:]
        psi_t2 = f['psi_t_r'][:] + 1.0j*f['psi_t_i'][:]
    
        prob_1p_output = probability_pnum_sites(output_sites,1,psi_t2,state_labels2)
        #ns_output_t2 = np.sum(ns_t2[1:,:],0)
        #ns_out_avg = np.average(ns_output_t2)
        #ns_output_fft = np.fft.fft(ns_output_t)
    
        #peaks,_ = find_peaks(np.abs(ns_output_fft))
    
        #signal_1p_avg.append(ns_out_avg)
        signal_1p_avg.append(np.average(prob_1p_output))
        
        f.close()

if figure_type == "rAB_ON_OFF_contrast" or figure_type == "time_rAB":
    rABs = np.asarray(rABs)
    rABs *= -1
    Us = np.asarray(Us)
    s_times = np.asarray(s_times)
    contrasts = np.asarray(contrasts)
    signal_2p_max = np.asarray(signal_2p_max)
    sort_idxs = np.argsort(rABs)
    rABs = rABs[sort_idxs]
    Us = Us[sort_idxs]
    s_times = s_times[sort_idxs]
    signal_2p_max = signal_2p_max[sort_idxs]

    rABs_theory = np.linspace(0.3,rABs[-1]*1.01,100)
    t12 = Us[0]*np.sqrt(2)*rABs_theory**2/(1+rABs_theory**2)**2
    delta = Us[0]*rABs_theory**2/(1+rABs_theory**2)**2
    signal_2p_max_theory = t12**2/((delta/2)**2+t12**2)

    if figure_type == "rAB_ON_OFF_contrast":
        rABs2 = -np.asarray(rABs2)
        signal_1p_avg = np.asarray(signal_1p_avg)
        sort_idxs2 = np.argsort(rABs2)
        rABs2 = rABs2[sort_idxs2]
        signal_1p_avg = signal_1p_avg[sort_idxs2]
        signal_1p_avg_theory = (2*rABs_theory**2+0.5)/((rABs_theory**2+1)**2)
        contrast = signal_2p_max/signal_1p_avg
        contrast_theory = signal_2p_max_theory/signal_1p_avg_theory

    Omegas_theory = (3/2)*Us[0]*rABs_theory**2/((1+rABs_theory**2)**2)
    print(Omegas_theory)
    s_times_theory = 0.5*np.pi/Omegas_theory


if figure_type == "rAB_ON_OFF_contrast":
    fig, (ax1,ax2) = plt.subplots(nrows=2,sharex=True)
    fig.set
    ax1.plot(rABs,s_times,label='Data',marker="o",linestyle='None')
    ax1.plot(rABs_theory,s_times_theory,label=r"Theory", linestyle="-",marker="None",color=bcolors[2])
    ax1.legend()
    
    ax1.set_ylabel(r"Switching time (1/$t_{\mathrm{AC}}$)")

    ax2.plot(rABs,contrast,label='Data',marker="o",linestyle="None")
    ax2.plot(rABs_theory,contrast_theory,label='Theory',marker="None",linestyle="-")
    ax2.set_xlabel("$r_{\mathrm{AB}}$")
    ax1.set_ylabel("Switching time $(1/t_{\mathrm{AC}})$")
    ax2.set_ylabel("ON/OFF Contrast")

    fig, ax = plt.subplots()
    ax.plot(rABs,signal_1p_avg,label='OFF, Data',marker='o',linestyle='None')
    ax.plot(rABs,signal_2p_max,label='ON, Data',marker='o',linestyle='None')
    ax.plot(rABs_theory,signal_1p_avg_theory,label='OFF, Theory',marker='None',linestyle='-')
    ax.plot(rABs_theory,signal_2p_max_theory,label='ON, Theory',marker='None',linestyle='-')

    ax.set_xlabel("$r_{\mathrm{AB}}$")
    ax.set_ylabel("ON/OFF Contrast")
    ax.legend()

    fig, (ax,ax1) = plt.subplots(nrows=2,sharex=True)
    ax.plot(rABs,signal_2p_max,label='ON, Data',marker='o',linestyle='None',color=bcolors[0])
    ax.plot(rABs_theory,signal_2p_max_theory,label='ON, Theory',marker='None',linestyle='-',color=bcolors[1])
    ax.plot(rABs,signal_1p_avg,label='OFF, Data',marker='o',linestyle='None',color=bcolors[2])
    ax.plot(rABs_theory,signal_1p_avg_theory,label='OFF, Theory',marker='None',linestyle='-',color=bcolors[3])

    ax.set_ylabel("Probability")
    ax.legend(frameon=False,loc='center right')

    ax1.plot(rABs,signal_2p_max/signal_1p_avg,label='Data',marker='o',linestyle='None',color=bcolors[0])
    ax1.plot(rABs_theory,signal_2p_max_theory/signal_1p_avg_theory,label='Theory',marker='None',linestyle='-',color=bcolors[3])
    ax1.set_xlabel("$r_{\mathrm{AB}}$")
    ax1.set_ylabel("ON/OFF")
    ax1.legend(frameon=False,loc='upper left')


    plt.show()


if figure_type == "time_rAB":
    fig, ax1 = plt.subplots()
    fig.set_figheight(figure_height/2*1.2)
    ax1.plot(rABs,s_times,label='Data',marker="o",linestyle='None',color=bcolors[0])
    ax1.plot(rABs_theory,s_times_theory,label=r"Theory", linestyle="-",marker="None",color=bcolors[3])
    ax1.legend(frameon=False)
    
    ax1.set_xlabel(r"$U$ ($t_{\mathrm{AC}}$)")
    ax1.set_ylabel(r"$t_{\mathrm{switch}}$ (1/$t_{\mathrm{AC}}$)")


    plt.show()

if figure_type == "time_U":
    Us = np.asarray(Us)
    s_times = np.asarray(s_times)
    sort_idxs = np.argsort(Us)
    Us = Us[sort_idxs]
    if lattice_name == "three_site":
        rABs = np.asarray(rABs)
        rABs *= -1
        rABs = rABs[sort_idxs]
    s_times = s_times[sort_idxs]


    Us_small = np.linspace(Us[0]*0.9,4,100)
    Us_large = np.linspace(9,Us[-1]*1.1,100)

    if lattice_name == "three_site":
        rAB = rABs[0]
        Omegas_U_small_theory = (3/2)*Us_small*rAB**2/((1+rAB**2)**2)
        Omegas_U_large_theory = (3/2)/Us_large

        s_times_U_small_theory = 0.5*np.pi/Omegas_U_small_theory
        s_times_U_large_theory = 0.5*np.pi/Omegas_U_large_theory

    if lattice_name == "diamond" and n_cells == 2:
        Omegas_U_small_theory = Us_small**2*6/(256*np.sqrt(2))
        s_times_U_small_theory = 0.5*np.pi/Omegas_U_small_theory

    if lattice_name == "diamond" and n_cells == 1:
        Omegas_U_small_theory = Us_small/8
        Omegas_U_large_theory = 2/Us_large
        s_times_U_small_theory = 0.5*np.pi/Omegas_U_small_theory

    fig, ax = plt.subplots()
    fig.set_figheight(figure_height/2*1.2)
    ax.loglog(Us,s_times,label='Data',marker='o',linestyle='None',color=bcolors[3])
    ax.set_xticks([1.0e-2,1.0e-1,1.0e0,1.0e1,1.0e2],minor=True)

    if lattice_name == "three_site":
        ax.loglog(Us_small,s_times_U_small_theory,label='Small $U$ theory',marker='None',linestyle='-',color=bcolors[0])
        ax.loglog(Us_large,s_times_U_large_theory,label='Large $U$ theory',marker='None',linestyle='-',color=bcolors[2])
        ax.set_xlabel('$U (t_{\mathrm{AC}})$')
        ax.set_ylabel('$t_{\mathrm{switch}}$ ($1/t_{\mathrm{AC}}$)')

    if lattice_name == "diamond" and n_cells == 2:
        ax.loglog(Us_small,s_times_U_small_theory,label='Small $U$ theory',marker='None',linestyle='-')
        ax.set_xlabel('$U$ (t)')
        ax.set_ylabel('$t_{\mathrm{switch}}$ ($1/t$)')

    if lattice_name == "diamond" and n_cells == 1:
        ax.loglog(Us_small,s_times_U_small_theory,label='Small $U$ theory',marker='None',linestyle='-')
        #ax.loglog(Us_large,s_times_U_large_theory,label='Large $U$ theory',marker='None',linestyle='-')
        ax.set_xlabel('$U$ (t)')
        ax.set_ylabel('$t_{\mathrm{switch}}$ ($1/t$)')

    ax.legend(frameon=False,loc='upper right')

    plt.show()


