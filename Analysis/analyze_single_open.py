import sys
import numpy as np
import matplotlib.pyplot as plt # plotting library
from matplotlib import colors
import matplotlib as mpl
import re


import h5py


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

import cmasher as cmr
import colorcet as cc

import distinctipy

def n_particle_indices(idxs,labels):
    n_particle_indices = {}
    for count, idx in enumerate(idxs):
        site_str = re.sub(r'[^0-9]','',labels[count])
        label_list = [*site_str]
        print(label_list)

        site_nums = [int(s) for s in label_list]
        pnum_sum = 0
        for n in site_nums: pnum_sum += n 
        print(pnum_sum)
        if pnum_sum not in n_particle_indices.keys():
            n_particle_indices[pnum_sum] = []
        n_particle_indices[pnum_sum].append(idx)
        print(n_particle_indices)
    for key,elem in n_particle_indices.items():
        elem.sort()
    return n_particle_indices



logscale = False

if len(sys.argv) == 1:
    print("Too few arguments given. Exiting.")
    exit()

path = sys.argv[1]

f = h5py.File(path,'r')
lattice_name = f['lattice_name'].asstr()[...]
print(lattice_name)
init_type = f['init_type'].asstr()[...]
print(init_type)
Nb_max = f['Nb_max'][()]
ts = f['ts'][:]
start = ts[0]
stop = ts[-1]
steps = ts.shape[0]

U = f['U'][()]
rAB = 0
if lattice_name == 'three_site':
    rAB = f['rAB'][()]
flux = 0
if lattice_name == 'diamond':
    flux = f['flux'][()]
n_cells = 1
if lattice_name != 'three_site':
    n_cells = f['n_cells'][()]
gammaR1 = f['gammaR'][()]
rho_t = f['rho_t_r'][:]+1.0j*f['rho_t_i'][:]
ns = f['ns'][:]

n_sites = ns.shape[0]

state_ints = f['state_ints'][:]
state_idxs = f['state_idxs'][:]
state_labels = f['state_labels'].asstr()[...]
print(state_labels)
f.close()



fig1,ax1 = plt.subplots()
if logscale == True:
    pos = ax1.pcolormesh(ts,np.linspace(1,n_sites,n_sites),ns,norm=colors.LogNorm(vmin=np.amin(ns[:,-1]), vmax=Nb_max),cmap=cc.cm.fire,shading='auto',rasterized=True)
if logscale == False:
    pos = ax1.pcolormesh(ts,np.linspace(1,n_sites,n_sites),ns,norm=colors.Normalize(vmin=1e-7, vmax=Nb_max),cmap=cc.cm.fire,shading='auto',rasterized=True)
ax1.set_xlabel("Time")
ax1.set_ylabel("Site")
ax1.set_yticks(np.arange(1.0,n_sites+1,step=1.0))
cbar = fig1.colorbar(pos,ax=ax1,pad=0.01)
#cbar.set_label("Photon number")

dcolors = distinctipy.get_colors(Nb_max+1)
bcolors = [(27.0/256,158.0/256,119.0/256),\
        (217.0/256,95.0/256,2.0/256),\
        (117.0/256,112.0/256,179.0/256),\
        (231.0/256,41.0/256,138.0/256)]

fig3,ax3 = plt.subplots()
if lattice_name == "diamond" and n_cells > 1:
    ax3.plot(ts,np.sum(ns[0:3,:],0), color = bcolors[0], label="1+2+3")
    ax3.plot(ts,np.sum(ns[-3:,:],0), color = bcolors[1], label=str(n_sites-2)+"+"+str(n_sites-1)+"+"+str(n_sites))
    ax3.plot(ts,np.sum(ns,0), color = bcolors[3], label ="Total")
elif lattice_name == "diamond" and n_cells == 1:
    ax3.plot(ts,ns[0,:],color = bcolors[0], label ="Left edge site")
    ax3.plot(ts,np.sum(ns[1:-1,:],0),color = bcolors[1], label ="Middle sites")
    ax3.plot(ts,ns[-1,:],color = bcolors[2], label ="Right edge site")
    ax3.plot(ts,np.sum(ns,0), color = bcolors[3], label ="Total")
elif lattice_name == "sawtooth" and n_cells > 1:
    ax3.plot(ts,np.sum(ns[0:2,:],0), color = bcolors[0], label="1+2")
    ax3.plot(ts,np.sum(ns[-2:,:],0), color = bcolors[1], label=str(n_sites-1)+"+"+str(n_sites))
    ax3.plot(ts,np.sum(ns,0), color = bcolors[3], label ="Total")
elif lattice_name == "sawtooth" and n_cells == 1:
    ax3.plot(ts,ns[0,:],color = bcolors[0], label ="Left edge site")
    ax3.plot(ts,ns[1,:],color = bcolors[1], label ="Middle site")
    ax3.plot(ts,ns[-1,:],color = bcolors[2], label ="Right edge site")
    ax3.plot(ts,np.sum(ns,0), color = bcolors[3], label ="Total")
elif lattice_name == "three_site":
    ax3.plot(ts,ns[0,:],color = bcolors[0], label ="Site 1")
    ax3.plot(ts,ns[1,:],color = bcolors[1], label ="Site 2")
    ax3.plot(ts,ns[-1,:],color = bcolors[2], label ="Site 3")
    ax3.plot(ts,np.sum(ns,0), color = bcolors[3], label ="Total")
if logscale == True:
    ax3.set_yscale('log')
#ax3.plot(ts,np.sum(ns[0:2,:],0)/Nb)
ax3.set_xlim([start,stop])
ax3.set_ylim([1e-6,Nb_max+0.05])
ax3.set_xlabel("Time")
ax3.set_ylabel("Photon number")
ax3.legend()

#fig3.savefig(fname = "./Figures/pnum_lr_time_"+output_name+".png",format = "png",bbox_inches="tight")
#fig3.savefig(fname = "./Figures/pnum_lr_time_"+output_name+".pdf",format = "pdf",bbox_inches="tight")

# Fix these!
#fig1.savefig("./Figures/"+name+"pnum_vs_site_time_"+output_name+".svg",format='svg')
#fig3.savefig("./Figures/"+name+"pnum_vs_time_"+output_name+".svg",format='svg')

#fig3,ax3 = plt.subplots()
#ax3.plot(ts,np.sum(ns,0),label="Total particle number")
#ax3.set_xlim([start,stop])
#ax3.set_xlabel("Time")
#ax3.set_ylabel("Particle number")
#ax3.legend()

print(np.polyfit(ts,np.log(np.sum(ns,0)),1))



fig3,ax3 = plt.subplots()
#ax3.plot(ts[0:-1],-(np.sum(ns,0)[1:]-np.sum(ns,0)[0:-1])/(ts[1]-ts[0]),label="tot pnum time-derivative")
#ax3.plot(ts,-np.gradient(np.sum(ns,0),ts[1]-ts[0]),label="tot pnum time-derivative")
Pns = np.zeros([Nb_max+1,steps])
indices_dict = n_particle_indices(state_idxs,state_labels)
print(indices_dict)
for n in range(Nb_max+1):
    if n not in indices_dict.keys():
        continue
    indices = indices_dict[n]
    rho_sum = np.trace(rho_t[indices,:,:][:,indices,:])
    print(rho_sum.shape)
    Pns[n,:] = rho_sum
    if n == 1:
        text = str(n)+ " Photon"
    else:
        text = str(n)+ " Photons"
    ax3.plot(ts,np.real(rho_sum),color=dcolors[n],label=text)
    if logscale:
        ax3.set_yscale('log')
    print(n)
    print(np.polyfit(ts[300:400],np.log(np.real(rho_sum[300:400])),1))
ax3.set_xlim([start,stop])
ax3.set_ylim([1e-7,1.05])
ax3.set_xlabel("Time")
ax3.set_ylabel("Probability")
ax3.legend()

# Fix this!
#fig3.savefig("./Figures/"+name+"_n_part_prob_vs_time_"+output_name+".svg",format='svg')

#fig4,ax4 = plt.subplots()
#dcolors = distinctipy.get_colors(basis.Ns)
#for n in range(basis.Ns):
#    if logscale:
#        ax4.semilogy(ts,np.real(rho_t[n,n,:]),color=dcolors[n],label=basis.int_to_state(basis[n]))
#    else:
#        ax4.plot(ts,np.real(rho_t[n,n,:]),color=dcolors[n],label=basis.int_to_state(basis[n]))
#
#
#ax4.set_xlim([start,stop])
#ax4.set_ylim([1e-7,1.05])
#ax4.set_xlabel("Time")
#ax4.set_ylabel("Probability")
#ax4.legend()

#fig5,ax5 = plt.subplots()
#for m in range(Nb_max+1):
#    if Pns[m,0] > 1e-5:
#        ax5.plot(ts,Es[m,:]/Pns[m,:],label=str(m)+" particles")
#ax5.set_xlabel("Time")
#ax5.set_ylabel("Energy expectation value/probability")
#ax5.legend()
#
#fig6,ax6 = plt.subplots()
#for m in range(Nb_max+1):
#    ax6.plot(ts,np.sum(Es,0),label="Total energy")
#ax5.set_xlabel("Time")
#ax5.set_ylabel("Energy expectation value")
#ax5.legend()

if len(sys.argv) > 2:
    path2 = sys.argv[2]
    f2 = h5py.File(path2,'r')
    lattice_name2 = f2['lattice_name'].asstr()[...]
    init_type2 = f2['init_type'].asstr()[...]
    Nb_max2 = f2['Nb_max'][()]
    ts2 = f2['ts'][:]
    U2 = f2['U'][()]
    rAB2 = 0
    if lattice_name == 'three_site':
        rAB2 = f2['rAB'][()]
    flux2 = 0
    if lattice_name == 'diamond':
        flux2 = f2['flux'][()]
    n_cells2 = 1
    if lattice_name != 'three_site':
        n_cells2 = f2['n_cells'][()]
    gammaR12 = f2['gammaR'][()]
    rho_t2 = f2['rho_t_r'][:]+1.0j*f2['rho_t_i'][:]
    ns2 = f2['ns'][:]
    n_sites2 = ns2.shape[0]
    
    state_ints2 = f2['state_ints'][:]
    state_idxs2 = f2['state_idxs'][:]
    state_labels2 = f2['state_labels'].asstr()[...]
    f2.close()

    Pns2 = np.zeros([Nb_max2+1,steps])
    indices_dict2 = n_particle_indices(state_idxs2,state_labels2)
    print(indices_dict2)
    for n in range(Nb_max2+1):
        if n not in indices_dict.keys():
            continue
        indices = indices_dict2[n]
        rho_sum = np.trace(rho_t2[indices,:,:][:,indices,:])
        Pns2[n,:] = rho_sum

    fig1,(ax1,ax2) = plt.subplots(nrows = 2, ncols=1, sharex = True)
    if logscale == True:
        pos1 = ax1.pcolormesh(ts,np.linspace(1,n_sites,n_sites),ns/Nb_max,norm=colors.LogNorm(vmin=np.amin(ns[:,-1]/Nb_max), vmax=1),cmap=cc.cm.fire,shading='auto',rasterized=True)
        pos2 = ax2.pcolormesh(ts2,np.linspace(1,n_sites2,n_sites2),ns2/Nb_max2,norm=colors.LogNorm(vmin=np.amin(ns2[:,-1]/Nb_max2), vmax=1),cmap=cc.cm.fire,shading='auto',rasterized=True)
    if logscale == False:
        pos1 = ax1.pcolormesh(ts,np.linspace(1,n_sites,n_sites),ns/Nb_max,norm=colors.Normalize(vmin=1e-7, vmax=1),cmap=cc.cm.fire,shading='auto',rasterized=True)
        pos2 = ax2.pcolormesh(ts2,np.linspace(1,n_sites2,n_sites2),ns2/Nb_max2,norm=colors.Normalize(vmin=np.amin(ns2[:,-1]/Nb_max2), vmax=1),cmap=cc.cm.fire,shading='auto',rasterized=True)
    #ax1.set_xlabel("Time")
    fig1.supylabel("Site")
    if lattice_name == 'three_site':
        fig1.supxlabel("Time (1/$t_{AC}$)")
    if lattice_name == 'sawtooth':
        fig1.supxlabel("Time (1/$t_{AA}$)")
    if lattice_name == 'diamond':
        fig1.supxlabel("Time (1/$t$)")
    ax1.set_yticks(np.arange(1.0,n_sites+1,step=1.0))
    if lattice_name == 'three_site':
        ax1.set_yticklabels(["A","B","C"])
    cbar = fig1.colorbar(pos1,ax=[ax1,ax2],pad=-0.03)

    ax2.set_yticks(np.arange(1.0,n_sites+1,step=1.0))
    if lattice_name == 'three_site':
        ax2.set_yticklabels(["A","B","C"])

    fig2,(ax21,ax22) = plt.subplots(nrows = 2,sharex=True)
    for n in reversed(range(Nb_max+1)):
        ax21.plot(ts,Pns[n],label="$P_{"+str(n)+"}$",color=bcolors[n])
    for n in reversed(range(Nb_max2+1)):
        ax22.plot(ts2,Pns2[n],label="$P_{"+str(n)+"}$",color=bcolors[n])
    ax21.legend(loc='center right',frameon=False)
    ax22.legend(loc='center right',frameon=False)
    ax1.set_yticks(np.arange(1.0,n_sites+1,step=1.0))
    if lattice_name == 'three_site':
        fig2.supxlabel("Time (1/$t_{AC}$)")
    if lattice_name == 'sawtooth':
        fig2.supxlabel("Time (1/$t_{AA}$)")
    if lattice_name == 'diamond':
        fig2.supxlabel("Time (1/$t$)")
    #fig2.supylabel("Probability")
    ax21.set_ylabel('Probability')
    ax22.set_ylabel('Probability')



    # Figure 3: comparison between interacting and non-interacting


plt.show()
