[![DOI](https://zenodo.org/badge/50950512.svg)](https://zenodo.org/badge/latestdoi/50950512)
[![Build Status](https://travis-ci.org/dafer45/TBTK.svg?branch=master)](https://travis-ci.org/dafer45/TBTK)

# TBTK
Welcome to TBTK, a library for modeling and solving second quantized Hamiltonians with discrete indices.
That is, Hamiltonians of the form  
<p align="center"><img src="doc/Hamiltonian.png" /></p>  
TBTK itself originated as a Tight-Binding ToolKit, and currently have complete support for modeling and solving the first bilinear term.
However, the scope of TBTK has expanded vastly since its inception.
It is today more generally a collection of data structures that are meant to enable rapid development of new algorithms for both interacting and non-interacting systems.
Examples include general purpose data structures for quantities such as the Density, DOS, EigenValues, (spin-polarized) LDOS, Magnetization, WaveFunctions, etc.
In addition to providing native solvers, TBTK thereby also aims to enable the development of frontends and backends to already existing packages that allows for seamless integration of the codebase already developed by the scientific community.
To aid such integration, TBTK is specifically designed to allow for solution algorithms to be used interchangably with minimal amount of modification of the code.  
<br/><br/>

**Full documenation:** http://www.second-quantization.com/  
**Also see:** http://www.second-tech.com

# Example
Consider the tight-binding Hamiltonian  
<p align="center"><img src="doc/ExampleHamiltonian.png" /></p>  

on a square lattice of size 30x30, where angle brackets denotes summation over nearest neighbors, and *sigma* is a spin summation index.
The parameters *t = 1 eV* and *J = 0.5 eV* are the nearest neighbor hopping amplitude and the strength of the Zeeman term, respectively.
Moreover, let the chemical potential be *mu = -1 eV*, the temperature be *T = 300K*, and the particle have Fermi-Dirac statistics.  
```cpp
const int SIZE_X                = 30;
const int SIZE_Y                = 30;
const double t                  = 1;
const double J                  = 0.5;
const double T                  = 300;
const Statistics statistics     = Statistics::FermiDirac;
```

Now assume that we are interested in calculating the density of states (DOS) and magnetization for the system.
For the DOS we want to use the energy window [-10, 10] and an energy resolution of 1000 points.  
```cpp
const double LOWER_BOUND        = -10;
const double UPPER_BOUND        = 10;
const int RESOLUTION            = 1000;
```
In addition we decide that the appropriate solution method for the system is diagonalization.
We proceed as follows.

## Setup the model  
```cpp
Model model;
for(int x = 0; x < SIZE_X; x++){
        for(int y = 0; y < SIZE_Y; y++){
                for(int s = 0; s < 2; s++){
                        //Add nearest neighbor hopping (HC for Hermitian conjugate).
                        if(x+1 < SIZE_X)
                                model << HoppingAmplitude(-t, {x+1, y, s}, {x, y, s}) + HC;
                        if(y+1 < SIZE_Y)
                                model << HoppingAmplitude(-t, {x, y+1, s}, {x, y, s}) + HC;

                        //Add Zeeman term.
                        model << HoppingAmplitude(-J*2*(1/2. - s), {x, y, s}, {x, y, s});
                }
        }
}
//Create Hilbert space basis.
model.construct();

//Set the chemical potential, temperature, and statistics.
model.setChemicalPotential(mu);
model.setTemperature(T);
model.setStatistics(statistics);
```

## Select solution method  
```cpp
Solver::Diagonalizer solver;
solver.setModel(model);
solver.run();
```

## Calculate properties  
```cpp
PropertyExtractor::Diagonalizer propertyExtractor(solver);
propertyExtractor.setEnergyWindow(LOWER_BOUND, UPPER_BOUND, RESOLUTION);

//Calculate the DOS.
Propery::DOS dos = propertyExtractor.calculateDOS();

//Calculate the Magnetization for all x and y values by passing the wildcard
//___ in the correpsonding positions. IDX_SPIN is used to tell the
//PropertyExtractor which subindex that corresponds to spin.
Property::Magnetization magnetization
        = propertyExtractor.calculateMagnetization({{___, ___, IDX_SPIN}});
```

## Plot and print results  
The DOS is a one-dimensional function of the energy and can easily be plotted.
We here do so using a Gaussian smoothin of 0.07.  
```cpp
Plotter plotter;
plotter.setLabelX("Energy");
plotter.setLabelY("DOS");
plotter.plot(dos, 0.07);
plotter.save("figures/DOS.png");
```
**Result:**
<p align="center"><img src="doc/DOS.png" /></p>  

For each point (x, y) on the lattice, the magnetization is a two-by-two complex matrix called a SpinMatrix.
The up and down components of the spin are given by the two diagonal entries.
We can therefore print the magnetization (the real part of the difference between the up and down components) at site (10, 10) as follows.
```cpp
SpinMatrix m = magnetization({10, 10, IDX_SPIN});
Streams::out << "Magnetization:\t" << real(m.at(0, 0) - m.at(1, 1)) << "\n";
```
**Result:**
```bash
Magnetization:	0.144248
```

For more examples and complete applications, see http://second-tech.com/wordpress/index.php/tbtk/ and the templates in the Templates folder.

# System requirements
**Verified to work with:**  
* gcc (v4.9 and up)
* clang (exact version number not known at the moment).  

**Required software and libraries:**  
* CMake (https://cmake.org/)
* BLAS (http://www.netlib.org/blas/)
* LAPACK (http://www.netlib.org/lapack/)

**Optional libraries:**
* ARPACK
* CUDA
* cURL
* FFTW3
* HDF5
* OpenCV
* OpenBLAS
* OpenMP
* SuperLU
* wxWidgets.

# Quickstart
## Installation
```bash
git clone http://github.com/dafer45/TBTK
mkdir TBTKBuild
cd TBTKBuild
cmake ../TBTK
make
sudo make install
```

## Run unit tests (optional)
```bash
make test
```

## Create a first application
```bash
cd ..
mkdir TBTKApplications
cd TBTKApplications
TBTKCreateAppliaction MyFirstApplication
cd MyFirstApplication
```

## Build and run the application
```bash
cmake .
make
./build/Application
```

# License
TBTK is free software that is licensed under the Appache 2.0 license (see the file
"License").  Please give attribution in accordance with the "Cite" section below.

### Third party license
#### json/TBTK/json.hpp (for serialization)
A thrid-party library hosted at https://github.com/nlohmann/json and is licensed under
the MIT license.


# Cite
#### To cite TBTK, mention TBTK in the text and cite the Digital Object Identifier (DOI) for this project.
[![DOI](https://zenodo.org/badge/50950512.svg)](https://zenodo.org/badge/latestdoi/50950512)  
Kristofer Björnson, & Andreas Theiler. (2017). dafer45/TBTK: First version with
native plotting [Data set]. Zenodo. http://doi.org/10.5281/zenodo.556398

#### If you use the ChebyshevSolver to produce results, please also cite the following references:
A. Weiße, G. Wellein, A. Alvermann, and H. Fehske,  
Rev. Mod. Phys. 78, 275 (2006).

L. Covaci, F. M. Peeters, and M. Berciu,  
Phys. Rev. Lett. 105, 167006 (2010).
