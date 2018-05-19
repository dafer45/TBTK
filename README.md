[![DOI](https://zenodo.org/badge/50950512.svg)](https://zenodo.org/badge/latestdoi/50950512)
[![Build Status](https://travis-ci.org/dafer45/TBTK.svg?branch=master)](https://travis-ci.org/dafer45/TBTK)

# TBTK
Welcome to TBTK, a library for modeling and solving second quantized Hamiltonians with discrete indices.
That is, Hamiltonians of the form  
<p align="center"><img src="doc/Hamiltonian.png" /></p>  
TBTK itself originated as a Tight-Binding ToolKit, and currently have complete support for modeling and solving the first bilinear term.
However, the scope of TBTK has expanded vastly since its inception.
It is today first and foremost a collection of data structures that are meant to enable rapid development of new algorithms for both interacting and non-interacting systems.
Examples of such data structures are quantities such as the Density, DOS, (spin-polarized) LDOS, Green's functions, Susceptibilities, etc.
TBTK thereby aims to enable the development of frontends and backends to already existing packages that allows for seamless integration of the codebase already developed by the scientific community.
To aid such integration, TBTK is specifically designed to allow for solution algorithms to be used interchangably with minimal amount of modification of the code.

Full documentation is available at:  http://www.second-quantization.com/  
Also see: http://www.second-tech.com

# Example
Consider the tight-binding Hamiltonian  
<p align="center"><img src="doc/ExampleHamiltonian.png" /></p>  
on a square lattice of size 20x20, where angle brackets denotes summation over nearest neighbors, and *sigma* is a spin summation index.
The parameters *t = 1 eV* and *J = 0.25 eV* are the nearest neighbor hopping amplitude and strength of a Zeeman term, respectively.
Moreover, let the chemical potential be *mu = -1 eV*, the temperature be *T = 300K*, and the particle have Fermi-Dirac statistics.
The model can then be setup as follows

```cpp
const int SIZE_X        = 20;
const int SIZE_Y        = 20;
const double t          = 1;
const double J          = 0.25;
 
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

//Set the chemical potential and temperature.
model.setChemicalPotential(-1);
model.setTemperature(300);
model.setStatistics(Statistics::FermiDirac);
```

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

## What's next?
For further information, see http://second-tech.com/wordpress/index.php/tbtk/ and the templates in the Template folder.


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
