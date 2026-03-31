## MC and SMD for 2D $J_1-J_2$ Heisenberg Models: a high-performance C++ implementation with MKL-DFTI, MPI and OpenCV

## About this repository:

This repository contains C++ and Python codes developed during a two-year postdoctoral research project focused on the investigation of spin excitations in disordered magnetic systems and the magnetic properties of quasicrystals (QCs). The implementation combines Monte Carlo (MC) simulations with parallel tempering (parallelization with MPI) and semiclassical molecular dynamics (SMD) to study $J_1-J_2$ Heisenberg models on two-dimensional lattices;

The MC code supports both Heisenberg and Ising models. For the Heisenberg case, spin updates are performed using a heat-bath algorithm combined with microcanonical (overrelaxation) updates, enhancing sampling efficiency. For the Ising model, updates are carried out via the standard single-spin-flip Metropolis algorithm;

Regarding the SMD implementation, spin dynamics are numerically obtained by integrating the Heisenberg equations of motion in the classical limit, where they reduce to the LLG equations describing spin precession (without damping) in an effective local magnetic field. Time evolution is computed using the fourth-order Runge-Kutta (RK4) method, supplemented by an energy-correction scheme to ensure numerical stability over long simulations (see the associated publications for further details and references);

Available periodic geometries include square, triangular, Lieb, hexagonal, and Kagome lattices. Other geometries must be configured manually within the code or loaded at runtime, as in the case of the octagonal Ammann-Beenker QC approximants provided (see the directory *QCrystal_Data*);

For periodic systems, a $J_3$ exchange coupling is implemented through a specific parameter in the code. For the aforementioned QC approximants, this parameter instead corresponds to a $J_5$ exchange coupling. An external magnetic field applied along the $z$ axis can also be included, and exchange anisotropy is introduced by modifying the $S^{\\,z}-S^{\\,z}$ coupling factors, yielding an XXZ-type model. Also, a system with disorder due to lattice impurities can be obtained by setting the disorder ratio/fraction parameter in one of the configuration files;

For SMD simulations, MC-generated spin configurations (samples) are required, these are recorded for temperatures below a certain threshold (defined within the code) during the measurement stage. A specific input setting then defines the target binary file with the samples. The main output of the SMD simulations is the averaged dynamical spin structure factor (DSSF). This quantity is recorded for several frequency slices and wave vectors within the first Brillouin zone, as well as along a predefined path for varying frequencies;

Additionally, MC and SMD codes employ OpenCV functions to produce images of sampled spin configurations (including final configurations from both thermalization and measurement stages), as well as videos showing system evolution in MC time and real time. A lattice inspection feature is also implemented using OpenCV, allowing the user to interactively verify the neighbors of each lattice site;

Follow the instructions in *Instructions.txt* and the Python script *Set_Params.py* for a detailed guide on setting simulation parameters, configuring the algorithms, compiling/running the code, and analyzing the outputs;

- *Project Title:* Excitações de spin em sistemas de spin desordenados

- *Affiliation:* Instituto de Física da Universidade de São Paulo

- *Principal Investigator:* Rafael Marques Paes Teixeira

- *Project Supervisor:* Eric de Castro e Andrade

- *Funder:* São Paulo Research Foundation (FAPESP)

- *Funding opportunity number:* 2023/06682-8

- *Grant:* https://bv.fapesp.br/pt/pesquisador/726791/rafael-marques-paes-teixeira/

## Reseach outputs:

1) **Disorder-induced damping of spin excitations in $\textbf{Cr}$-doped $\textbf{BaFe}_{2}\textbf{As}_2$** (https://doi.org/10.1103/rkjn-hf7z)
   
   *Physical Review Research: 8, L012028*
   
   **Authors:** Marli R. Cantarino, Rafael M. P. Teixeira, K. R. Pakuszewski, Wagner R. da Silva Neto, Juliana G. de Abrantes, Mirian Garcia-Fernandez, P. G. Pagliuso, C. Adriano, Claude Monney, Thorsten Schmitt, Eric C. Andrade, Fernando A. Garcia

2) **Stripe order in quasicrystals** (https://doi.org/10.1140/epjb/s10051-025-01040-y)
   
   *European Physical Journal B: 98, 188*
   
   **Authors:** Rafael M. P. Teixeira, Eric C. Andrade

## About the model:

Heisenberg Hamiltonian with multiple exchange couplings, XXZ anisotropy, and an external magnetic field:

$H = \dfrac{1}{2} \sum_{\\,i,j,\mu} J_{ij}^{\\,\mu}\\, S_i^{\\,\mu} S_j^{\\,\mu} - h \sum_{\\,i} S_i^{\\,z}\\,$;

- $\text{Heisenberg model: }\vec{S}_i = (S_i^{\\,x}, S_i^{\\,y}, S_i^{\\,z})\text{ with }|\vec{S}_i| = 1\text{ (unit vector on the sphere)}\\,$;
- $\text{Ising model limit: }\vec{S}_i = S_i^{\\,z}\hat{z}\text{ with }S_i^{\\,z}= \pm 1\\,$;

Here, $\mu = x,y,z$ denotes spin components, $\vec{S} _ i$ is a classical spin representing a local magnetic moment on the site $i=1,2,\dots,N-1,N$ (with $N$ being the total number of sites), $J_{ij}^{\\,\mu}$ are exchange couplings such that $J_{ij}^{\\,x,y} = J_{ij}$ and $J_{ij}^{\\,z} = \lambda\\, J_{ij}$ with $\lambda$ controlling the exchange anisotropy, and $h$ is the external magnetic field along the $z$ direction;

For isotropic exchange couplings ($\lambda=1$), the Hamiltonian implemented in the code can be expressed as:

$H = H_1 + H_2 + H_f - h \sum_i S_i^{\\,z}\\,$;

- $H_1 = J_1 \sum_{\langle i,j \rangle} \vec{S}_i \cdot \vec{S}_j\text{ : nearest neighbors (NN);}$
- $H_2 = J_2 \sum_{\langle\langle i,j \rangle\rangle} \vec{S}_i \cdot \vec{S}_j\text{ : next-nearest neighbors (NNN);}$
- $H_f = J_f \sum_{\langle\langle\langle i,j \rangle\rangle\rangle} \vec{S}_i \cdot \mathbf{S}_j\text{ : further neighbors (3rd for crystals, 5th for QCs);}$

Here, $J_1$, $J_2$, $J_f$ (denoted as `JX` in the code) are exchange couplings, with $J_f = J_3$ (crystalline case) or $J_5$ (QC case);

## Semiclassical dynamics and DSSF:

In the semiclassical limit, the spins $\vec{S}(\vec{r} _ i , t) = \vec{S} _ {i}(t)$ evolve in time $t$ according to their precessional dynamics about the local effective field $\vec{h} _ {i}(t)$ (setting $\hbar=1$):

$\dfrac{d S_j^{\\,\mu}(t)}{dt} = i \\, [H,S_j^{\\,\mu}(t)] \\,\rightarrow\\, \dfrac{d \vec{S} _ {j}(t)}{dt} = \vec{h} _ {j}(t) \times \vec{S} _ {j}(t)\\,$;

$h_i^{\mu}(t) = \sum_j J_{ij}^{\\,\mu}\\, S_j^{\\,\mu}(t)\\,$;

The ensemble-averaged, momentum- and frequency-resolved dynamical spin correlation function is given by the classical DSSF:

$S^{\\,\mu\nu}(\vec{q}, \omega) = \dfrac{1}{2\pi N} \sum_{\\,ij} \int_{-\infty}^{\infty} dt \\, \langle S_i^{\\,\mu}(t) S_j^{\\,\nu}(0) \rangle \\, e^{\\,i(\\,\omega t - \vec{q} \cdot \vec{R} _ {\\,ij}\\,)}\\,$;

Here, $\vec{q}$ is a wave vector in reciprocal space, $\mu$ and $\nu$ denote spin components, and $\vec{R} _ {\\,ij} = \vec{r} _ i - \vec{r} _ j$ is a lattice displacement vector.
