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
To aid such seamless integration, TBTK is designed to allow for solution algorithms to be used interchangably with minimal amount of modification of the code.

Full documentation is available at:  http://www.second-quantization.com/  
Also see: http://www.second-tech.com

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
