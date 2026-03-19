# MC and SMD for 2D J1–J2 Heisenberg Models

### High-performance C++ implementation with OpenMPI and OpenCV

This repository contains C++ and Python codes developed during a two-year postdoctoral research project focused on the investigation of spin excitations in disordered magnetic systems and the magnetic properties of quasicrystals (QCs). The implementation combines Monte Carlo (MC) simulations with parallel tempering (parallelization with OpenMPI) and semiclassical molecular dynamics (SMD) to study J1–J2 Heisenberg and Ising models on two-dimensional lattices;

For Heisenberg models, spin updates are carried out using the heat-bath algorithm, while for Ising models, the standard single-spin-flip Metropolis algorithm is employed. In the SMD framework , spin dynamics are numerically obtained by integrating the Heisenberg equations of motion in the classical limit , where they reduce to the LLG equations describing spin precession (without damping) in an effective magnetic field. The fourth-   order Runge–Kutta (RK4) method is employed and supplemented by an energy-correction scheme, ensuring stable numerical evolution (see the published    articles for more details and references);

Available periodic geometries include square, triangular, Lieb, hexagonal, and Kagome lattices. Other geometries must be configured manually within the code or loaded at runtime, as in the case of the Ammann–Beenker QC approximants provided (see the directory *QCrystal_Data*);

For periodic systems, a J3 exchange coupling is implemented through the parameter JX. For the aforementioned QC approximants, this parameter instead corresponds to a J5 exchange coupling. An external (z-axis) magnetic field can be included by setting a finite value for the corresponding parameter;

Interaction anisotropy can be introduced by setting the Sz–Sz coupling factors to realize an XXZ model. A system with disorder due to lattice impuri  ties can be obtained by setting the disorder ratio (or fraction) parameter in one of the configuration files (if set to 0, the system is clean);

For SMD simulations, MC–generated spin configurations (samples) recorded at temperatures below a certain threshold (defined within the code) during the measurement stage are required. A specific input configuration file then defines the target sample file;

The main output of the SMD simulations is the averaged dynamical structure factor (SF). This quantity is recorded for several frequency slices & wave vectors within the first Brillouin zone, as well as along a predefined path for varying frequencies;

Additionally, MC and SMD codes employ OpenCV functions to produce images of sampled spin configurations (including final configurations from both thermalization and measurement stages), as well as videos showing system evolution in MC time and real time. A lattice inspection feature is also implemented using OpenCV, allowing the user to interactively verify the neighbors of each lattice site; 

- Project Title: Excitações de spin em sistemas de spin desordenados

- Affiliation: Instituto de Física da Universidade de São Paulo

- Principal Investigator: Rafael Marques Paes Teixeira

- Project Supervisor: Eric de Castro e Andrade

- Funder: São Paulo Research Foundation (FAPESP)

- Funding opportunity number: 2023/06682-8

- Grant: https://bv.fapesp.br/pt/pesquisador/726791/rafael-marques-paes-teixeira/

## Reseach outputs:

1) Disorder-induced damping of spin excitations in Cr-doped BaFe2As2 (https://doi.org/10.1103/rkjn-hf7z)
   
   *Physical Review Research: 8, L012028*
   
   **Authors:** Marli R. Cantarino, Rafael M. P. Teixeira, K. R. Pakuszewski, Wagner R. da Silva Neto, Juliana G. de Abrantes, Mirian Garcia-Fernandez, P. G. Pagliuso, C. Adriano, Claude Monney, Thorsten Schmitt, Eric C. Andrade, Fernando A. Garcia

3) Stripe order in quasicrystals (https://doi.org/10.1140/epjb/s10051-025-01040-y)
   
   *European Physical Journal B: 98, 188*
   
   **Authors:** Rafael M. P. Teixeira, Eric C. Andrade

## About the model:

Heisenberg (or Ising) Hamiltonian with multiple exchange couplings and an external magnetic field on a two-dimensional lattice system:

$H = H_1 + H_2 + H_f - h \sum_i S_i^{\\,z}$;

- $H_1 = J_1 \sum_{\langle i,j \rangle} \vec{S}_i \cdot \vec{S}_j\text{ : nearest neighbors (NN);}$
- $H_2 = J_2 \sum_{\langle\langle i,j \rangle\rangle} \vec{S}_i \cdot \vec{S}_j\text{ : next-nearest neighbors (NNN);}$
- $H_f = J_f \sum_{\langle\langle\langle i,j \rangle\rangle\rangle} \vec{S}_i \cdot \mathbf{S}_j\text{ : further neighbors (3rd for crystals, 5th for QCs);}$

Here, $\vec{S}_i$ denotes a classical spin representing a local magnetic moment on the site $i=1,2,\dots,N-1,N$ –with $N$ being the total number of sites – where:

- Heisenberg: $\vec{S}_i = (S_i^{\\,x}, S_i^{\\,y}, S_i^{\\,z})$, with $\|\vec{S}_i\| = 1$ (unit vector on the sphere);
- Ising: $S_i = S_i^{\\,z} = \pm 1$;

In the Hamiltonian above, $J_1$, $J_2$, $J_f$ (JX on code) are exchange couplings, with $J_f = J_3$ (crystalline case) or $J_5$ (quasicrystal case), and $h$ is the external magnetic field along the $z$-direction;

## Semiclassical dynamics & dynamical spin structure factor (DSSF):

In the semiclassical limit, the spins $\vec{S}(\vec{r} _ i , t) = \vec{S} _ {i}(t)$ evolve in time $t$ according to their precessional dynamics about the local effective field $\vec{h} _ {i}(t)$ (setting $\hbar=1$):

$\dfrac{d S_j^{\\,\mu}(t)}{dt} = i [H,S_j^{\\,\mu}(t)] \rightarrow \dfrac{d \vec{S} _ {j}(t)}{dt} = \vec{h} _ {j}(t) \times \vec{S} _ {j}(t)$;

$h_i^{\mu}(t) = \sum_j J_{ij}^{\\,\mu}\\, S_j^{\\,\mu}(t)$;

The ensemble-averaged, energy- and momentum-resolved dynamical spin correlation function is given by the classical DSSF:

$S^{\\,\mu\nu}(\vec{q}, \omega) = \dfrac{1}{2\pi N} \sum_{\\,ij} \int_{-\infty}^{\infty} dt \langle S_i^{\\,\mu}(t) S_j^{\\,\nu}(0) \rangle \\, e^{\\,i\\,(\omega t - \vec{q} \cdot \vec{R} _ {\\,ij})}$

Here, $\vec{q}$ is a wave vector in reciprocal space, $\mu$ and $\nu$ denote spin components, and $\vec{R} _ {\\,ij} = \vec{r} _ i - \vec{r} _ j$.
