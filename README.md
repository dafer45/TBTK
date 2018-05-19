[![DOI](https://zenodo.org/badge/50950512.svg)](https://zenodo.org/badge/latestdoi/50950512)
[![Build Status](https://travis-ci.org/dafer45/TBTK.svg?branch=master)](https://travis-ci.org/dafer45/TBTK)

Welcome to TBTK, a library for modeling and solving second quantized Hamiltonians.

Full documentation is available at:  http://www.second-quantization.com/


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
cd TBTKApplication
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
