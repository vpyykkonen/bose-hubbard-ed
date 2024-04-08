# Bose-Hubbard ED
Simulating dynamics of simple systems with interacting spin-zero particles in an exact basis. The basis is specified via a set of single-particle states (e.g. tight-binding model of lattice orbitals and interorbital connections) and the considered number of bosons. The codes allow simulating both closed dynamics (`bose_hubbard_ed_closed.py`) and open dynamics  with sources and sinks for particles (`bose_hubbard_ed_open.py`). Also, the mean-field, that is, classical apporoximation can be considered (`bose_hubbard_classical.py`) where in the Heisenberg equation of motion, the creation and annihilation operators are replaced by their mean values, that is, a classical number.
The Hilbert space basis is handled using [QuSpin](https://quspin.github.io/QuSpin/).
The codes were used to produce the numerical simulations and visualisations of [Pyykkönen et al. All-optical switching at the two-photon limit with interference-localized states, Phys. Rev. Research __5__, 043259, (2023)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.043259).
The given analyses are thus made for the paper.


## Setup
1. Install QuSpin following the instructions in [https://quspin.github.io/QuSpin/Installation.html](https://quspin.github.io/QuSpin/Installation.html)
2. If h5py, the Python interface for hdf5, is not installed (you can check by running `python -c "import h5py"` on command line), install it e.g. via Anaconda by running `conda install h5py`.
3. If you installed QuSpin through Anaconda, initialize the environment with `conda init` (if not already initialized).

## Running the codes
The full exact diagonalization codes are divided in two phases: 1) Generating data, 2) Analyzing data.
The _generation_ occurs via the Python scripts `bose_hubbard_ed_closed.py`,`bose_hubbard_ed_open.py`, which runs simulation on a single parameter set and saves the results in an hdf5 file. The scripts can be run either with command line parameters (as defined in them), or one can change the parameters in the files and run without the command line parameters. _Analysis_ occurs with scripts contained in the folder `Analysis`, where template analyses are given. Presently, there are analyses for single simulations: `analyze_single_closed.py` and `analyze_single_open.py`, which are run e.g. `python analyze_single_closed.py path_to_data_file.h5`. There are also two example scripts for analyzing sets of simulations, where specific parameters are varied: `analyze_dataset_closed.py` and `analyze_dataset_open.py`, which are run as e.g. `python analyze_dataset_eg.py path_to_data_folder`.

At the moment, the mean-field simulation and the respective analysis are done in the same Python script `bose_hubbard_classical.py`.

## Structure of the analyses
In the analysis, the dynamics of the systems are simulated from a given initial state until a final time.
The structure of the simulations are:
1. Define system parameters and simulations times
    * `Nb` (closed),`Nb_max` (open) = initial number of bosons
    * `U` = on-site interaction strength
    * `gammaL1`, `gammaL2`,`gammaR1`,`gammaR2` =losses and gains at left and right edge sites of the system (open)
    * `start, stop, num` = start and stop times of the simulation and number of time points
2. Define system Hamiltonian and the initial state. For each model one defines a lattice by entering an `if 'lattice_name' = model_name:` block that compiles the single-particle and interaction Hamiltonians and defines the initial states in internal `if 'init_state' = state_name:` blocks.
3. System assembly and time-evolution from the initial state is perfomed via QuSpin. 
4. Results are saved into a hdf5 file named as specified in the code as `<name>_<closed/open>_<init_type>_Nb<Nb>_U<U>`, where `<...>` represent instered string or value and `name` is specified in the definition block of the model. The saved parameters are
    * For closed simulation: lattice name `lattice_name`, initial state type `init_type`, interaction strength `U`, model specific parameters, Hamiltonian eigenvalues and real and imaginary parts of eigenvectors `eigenvalues`, `eigenvectors_r`, `eigenvectors_i`, real and imaginary parts of overlaps between eigenvectors and initial state `overlaps_r` and `overlaps_i`, number of sites `n_sites`, number of particles `Nb`, considered time instances `ts`, real and imaginary parts of state vectors at each time `psi_t_r` and `psi_t_i`, particle number at each site at each time `ns_t`, basis state integer, index and label representations `state_ints`, `state_idxs`, `state_labels`.
    * For open simulation: lattice name `lattice_name`, initial state type `init_type`, interaction strength `U`, model specific parameters, state `overlaps_r` and `overlaps_i`, number of sites `n_sites`, number of particles `Nb`, considered time instances `ts`, real and imaginary parts of state vectors at each time `psi_t_r` and `psi_t_i`, particle number at each site at each time `ns_t`, basis state integer, index and label representations `state_ints`, `state_idxs`, `state_labels`.

