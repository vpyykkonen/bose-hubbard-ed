import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import h5py
import sys
import math
import glob
from quspin.basis import boson_basis_1d # bosonic Hilbert space
from scipy.optimize import curve_fit



#SMALL_SIZE = 22
#SMALL_SIZE = 16
SMALL_SIZE = 13
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


import cmasher as cmr
import colorcet as cc

import distinctipy

bcolors = [(27.0/256,158.0/256,119.0/256),\
        (217.0/256,95.0/256,2.0/256),\
        (117.0/256,112.0/256,179.0/256),\
        (231.0/256,41.0/256,138.0/256)]

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
        indices.append(int(basis.index(state)))
    return indices

def find_nearest(array,value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]

def exp_func(x, a, b, c):
    return a*np.exp(b*x)+c

def power_func(x, a, b, c):
    return a*x**b+c

figure_type = sys.argv[1]
path = sys.argv[2]

files = glob.glob(path+"/*.h5")
rABs  = []
gammas = []
Us = []
s_times = []
prob_1p_U0_stime = []
threshold = 0.001
n_cells = 0

f = h5py.File(files[0],'r')
lattice_name = f['lattice_name'].asstr()[...]
n_sites = int(f['n_sites'][()])
Nb_max = int(f['Nb_max'][()])
if lattice_name == "diamond":
    n_cells = int(f['n_cells'][()])

f.close()

Nb = [n for n in range(Nb_max+1)]
print(Nb)
basis = boson_basis_1d(n_sites,Nb=Nb)
indices_2p = n_particle_indices(basis,2,n_sites)
decay_rates = []

count = 0
for file in files:
    f = h5py.File(file,'r')
    if lattice_name == "three_site":
        rABs.append(f['rAB'][()])
    gammas.append(f['gammaR'][()])
    Us.append(f['U'][()])
    ts = f['ts'][:]
    rho_t = f['rho_t_r'][:]+1.0j*f['rho_t_i'][:]
    count += 1
    print(file)
    print(count)
    print(rho_t.shape)
    rho_2p = np.real(np.trace(rho_t[indices_2p,:,:][:,indices_2p,:]))
    idx, val = find_nearest(rho_2p,threshold)
    s_times.append(ts[idx])
    f.close()


rABs = np.asarray(rABs)
gammas = np.asarray(gammas)
Us = np.asarray(Us)
s_times = np.asarray(s_times)
#decay_rates = np.asarray(decay_rates)
rABs *= -1

if figure_type == "rAB_ON_OFF_contrast" or figure_type =="time_rAB":
    sort_idxs = np.argsort(rABs)
    rABs = rABs[sort_idxs]
    Us = Us[sort_idxs]
    gammas = gammas[sort_idxs]
    s_times = s_times[sort_idxs]
if figure_type == "time_U":
    sort_idxs = np.argsort(Us)
    Us = Us[sort_idxs]
    if lattice_name == "three_site":
        rABs = rABs[sort_idxs]
    gammas = gammas[sort_idxs]
    s_times = s_times[sort_idxs]
if figure_type == "time_gamma":
    sort_idxs = np.argsort(gammas)
    gammas = gammas[sort_idxs]
    Us = Us[sort_idxs]
    if lattice_name == "three_site":
        rABs = rABs[sort_idxs]
    s_times = s_times[sort_idxs]



if len(sys.argv) == 4:
    path2 = sys.argv[3]
    
    files2 = glob.glob(path2+"/*.h5")
    rABs2  = []
    
    f2 = h5py.File(files2[0],'r')
    n_sites2 = int(f2['n_sites'][()])
    Nb_max2 = int(f2['Nb_max'][()])
    f2.close()

    Nb2 = [n for n in range(Nb_max2+1)]
    print(Nb2)
    basis2 = boson_basis_1d(n_sites2,Nb=Nb2)
    indices_1p = n_particle_indices(basis2,1,n_sites2)
    
    print(indices_1p)
    for count,file in enumerate(files2):
        f2 = h5py.File(file,'r')
        if lattice_name == "three_site":
            rABs2.append(f2['rAB'][()])
        #gammas2.append(f2['gammaR'][()])
        ts2 = f2['ts'][:]
        rho_t2 = f2['rho_t_r'][:]+1.0j*f2['rho_t_i'][:]
        rho_1p = np.real(np.trace(rho_t2[indices_1p,:,:][:,indices_1p,:]))
        stime_interp = np.interp(rABs2[count],rABs,s_times)
        stime_index,val = find_nearest(ts2,stime_interp)
        prob_1p_U0_stime.append(rho_1p[stime_index])
        #decay_rates.append(-np.polyfit(ts[1000:],np.log(rho_2p[1000:]),1)[0])
        f2.close()
    
    
    rABs2 = np.asarray(rABs2)
    #gammas2 = np.asarray(gammas2)
    #Us2 = np.asarray(Us2)
    prob_1p_U0_stime = np.asarray(prob_1p_U0_stime)
    rABs2 *= -1
    
    sort_idxs = np.argsort(rABs2)
    prob_1p_U0_stime = prob_1p_U0_stime[sort_idxs]

    #rABs = rABs[sort_idxs]
    #gammas = gammas[sort_idxs]
    


if figure_type == "rAB_ON_OFF_contrast":
    fig, (ax1,ax2) = plt.subplots(nrows=2,sharex=True)
    ax1.semilogy(rABs,s_times,label='Data',marker='o',linestyle="None",color=bcolors[0])
    #ax1.set_xlim(rABs[0],rABs[-1])
    #ax1.set_ylim(np.amin(s_times),np.amax(s_times))
    rABs_cont = np.linspace(4.5,rABs[-1],100)
    t12 = np.sqrt(2)*rABs_cont**2/(1+rABs_cont**2)**2
    s_time_theory = -2.0*gammas[0]*np.log(threshold)/(4*t12**2)
    ax1.semiloy(rABs_cont,s_time_theory,label='Theory',marker='None',linestyle="-",color=bcolors[3])
    #print("prob_1p_U0_stime",marker='None',linestyle="-")
    print(prob_1p_U0_stime)
    contrast = (1-threshold)/(1-prob_1p_U0_stime)
    contrast_theory = (1-threshold)*(rABs**2+1)
    ax2.plot(rABs,contrast,label='Data',marker='o',linestyle="None",color=bcolors[0])
    ax2.plot(rABs,contrast_theory,label='Theory',marker="None",linestyle="-",color=bcolors[3])
    #ax2.set_xlim(rABs[0],rABs[-1])
    #ax2.set_ylim(np.amin(contrast),np.amax(contrast))
    ax2.set_xlabel("$r_{\mathrm{AB}}$")
    ax1.set_ylabel("$t_{\mathrm{switch}}$ ($1/t_{\mathrm{AC}}$)")
    ax2.set_ylabel("$\mathrm{ON/OFF}$")
    ax1.legend()


if figure_type == "time_U":
    fig,ax = plt.subplots()
    fig.set_figwidth(figure_width)
    fig.set_figheight(2.8)
    ax.loglog(Us,s_times,label='Data',marker="o",linestyle="None",color=bcolors[0])
    if lattice_name == "three_site":
        ax.set_xlabel("U ($t_{AC}$)")
        ax.set_ylabel("$t_{\mathrm{switch}}$ (1/$t_{AC}$)")
        rAB = rABs[0]
    if lattice_name == "diamond":
        ax.set_xlabel("U ($J$)")
        ax.set_ylabel("$t_{\mathrm{switch}}$ (1/$J$)")
    gamma = gammas[0]
    Us_large_theor = np.linspace(25,Us[-1]*1.1,100)
    Us_small_theor = np.linspace(Us[0]*0.9,0.8,100)

    if lattice_name == "three_site":
        t12_smallU = np.sqrt(2)*rAB**2*Us_small_theor/(1+rAB**2)**2
        t12_largeU = np.sqrt(2)/Us_large_theor
    if lattice_name == "diamond":
        t12_smallU = Us_small_theor/8
        t12_largeU = 1/Us_large_theor



    s_times_small_U_theor = -2.0*gamma*np.log(threshold)/(4*t12_smallU**2)
    s_times_large_U_theor = -2.0*gamma*np.log(threshold)/(4*t12_largeU**2)
    ax.plot(Us_small_theor,s_times_small_U_theor,label='Small $U$ theory',marker="None",linestyle="-",color=bcolors[3])
    if lattice_name == "three_site":
        ax.plot(Us_large_theor,s_times_large_U_theor,label='Large $U$ theory',marker="None",linestyle="-",color=bcolors[2])
    ax.legend(frameon=False)

    # small U limit

if figure_type == "time_rAB":
    fig,ax = plt.subplots()
    ax.plot(rABs,s_times,label='Data')
    ax.set_xlabel("$r_{AB}$")
    ax.set_ylabel("Switching time (1/$t_{AC}$)")

if figure_type == "time_gamma":
    fig,ax = plt.subplots()
    fig.set_figwidth(figure_width)
    fig.set_figheight(2.8)
    ax.loglog(gammas,s_times,label='Data',color=bcolors[0],marker ='o',linestyle="None")
    if lattice_name == "three_site":
        ax.set_xlabel("$\gamma~(t_{AC})$")
        ax.set_ylabel("$t_{\mathrm{switch}}$ (1/$t_{AC}$)")
        rAB = rABs[0]
    if lattice_name == "diamond":
        ax.set_xlabel("$\gamma~(J)$")
        ax.set_ylabel("$t_{\mathrm{switch}}$ (1/$J$)")

    U = Us[0]

    gamma_large_theor = np.linspace(0.2,2,100)
    gamma_small_theor = np.linspace(gammas[0]*0.9,0.2,100)

    if lattice_name == "three_site":
        t12 = np.sqrt(2)*rAB**2*Us[0]/(1+rAB**2)**2
    if lattice_name == "diamond":
        t12 = U/8


    s_times_large_gamma_theor = -2.0*gamma_large_theor*np.log(threshold)/(4*t12**2)

    if lattice_name == "three_site":
        delta = Us[0]*rAB**2/(1+rAB**2)**2
    if lattice_name == "diamond":
        delta = 0
    #s_times_small_gamma_theor = -np.sqrt(2)/2*np.log(threshold)*(delta**2+4*t12**2)/(2*gamma_small_theor*t12**2)
    s_times_small_gamma_theor = -1.0/2*np.log(threshold)*(delta**2+4*t12**2)/(2*gamma_small_theor*t12**2)

    ax.loglog(gamma_small_theor,s_times_small_gamma_theor,label='Small $\gamma$',marker="None",linestyle="-",color=bcolors[2])
    ax.loglog(gamma_large_theor,s_times_large_gamma_theor,label='Large $\gamma$',marker="None",linestyle="-")
    ax.legend(loc='lower right',frameon=False)
    ax.set_xlim(gammas[0],90)

#ax.plot(gammas,s_times,label='Data')
#popt, pcov = curve_fit(exp_func, rABs[6:], s_times[6:])
#print(popt)
#ax.plot(rABs[8:], exp_func(rABs[8:],*popt), '--')

#popt, pcov = curve_fit(power_func, rABs[:], s_times[:])
#print(popt)
#ax.plot(rABs, power_func(rABs,*popt), '--',label='Power law fit')

#ax.plot(rABs,s_times,label='Data')
#ax.set_xlabel("rAB")

#ax.set_xlabel("gamma")


#print("Fits to the ends")
#print(np.polyfit(np.log(Us[0:10]),np.log(s_times[0:10]),1))
#print(np.polyfit(Us[-5:-1],np.log(s_times[-5:-1]),1))

#fig, ax = plt.subplots()
#ax.plot(rABs,s_times,label='Data')
#xs = np.linspace(rABs[0],rABs[-1],100)
#n = 4
#p = np.polyfit(rABs,s_times,n)
#print(p)
#ax.plot(xs,np.poly1d(p)(xs),'--',label="Polyfit, n = "+str(n))
#ax.legend()
#
#fig,ax = plt.subplots()
#ax.plot(rABs,decay_rates)
#
#

#Us2 = np.linspace(0.1,2.0,100)
#t12 = -0.0783
#gamma = gammas[0]
#delta = 0.05364
#factor = np.real(np.sqrt((1.0+0.0j)*(gamma**2-32*(t12*Us2)**2+16*(delta*Us2)**2)))
#stimes2 = (np.log(threshold)-np.log((factor+gamma)/(2*factor)))/(-0.25*(gamma-factor))
#
#ax.plot(Us2,stimes2,label='Two state model, large gamma')
#ax.legend()

#Us2 = np.linspace(0.1,2.0,100)
#t12 = -0.0783
#gamma = gammas[0]
#delta = 0.0
#factor = np.sqrt(gamma**2-32*(t12*Us2)**2+16*(delta*Us2)**2)
#stimes2 = (np.log(threshold)-np.log((factor+gamma)/(2*factor)))/(-0.25*(gamma-factor))
#
#ax.loglog(Us2,stimes2)


plt.show()


