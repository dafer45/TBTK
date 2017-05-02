[![DOI](https://zenodo.org/badge/50950512.svg)](https://zenodo.org/badge/latestdoi/50950512)

Welcome to the Tight Binding Toolkit - TBTK

For an introduction to the library, see http://dafer45.github.io/TBTK

# Installation instructions
Without CUDA (gpu)  
```bash
source init_session.sh  
./install.sh
```

With CUDA (gpu)  
```bash
source init_session.sh
./install -CUDA
```

# Update TBTK
After pulling the latest version of TBTK, execute the following command from
the TBTK root folder to update the library:

Without CUDA (gpu)
```bash
./update.sh
```

With CUDA (gpu)  
```bash
./update.sh -CUDA
```

# Initialize session
Each time a new terminal session is opened, execute the following command from
the TBTK root folder:  
```bash
source init_session.sh
```

# License
TBTK is free to use under the Appache 2.0 license (see the file 'License').
Please give attribution in accordance with the 'Cite' section below.

### Third party licenses
#### json/json.hpp (for serialization)
A thrid-party library hosted at https://github.com/nlohmann/json and is
licensed under the MIT license.
#### HDF5 (for saving data)
HDF5 is not distributed with TBTK, but is downloaded as part of the
installation procedure. After installation HDF5's BSD-style License can be
found in hdf5/hdf5-1.8.16/COPYING.



# Cite
#### To cite TBTK, mention TBTK in the text and cite the DOI for this project.  
[![DOI](https://zenodo.org/badge/50950512.svg)](https://zenodo.org/badge/latestdoi/50950512)  
Kristofer Björnson, & Andreas Theiler. (2017). dafer45/TBTK: First version with
native plotting [Data set]. Zenodo. http://doi.org/10.5281/zenodo.556398

#### If you use the ChebyshevSolver to produce results, please also cite the
following references  
A. Weiße, G. Wellein, A. Alvermann, and H. Fehske,  
Rev. Mod. Phys. 78, 275 (2006).

L. Covaci, F. M. Peeters, and M. Berciu,  
Phys. Rev. Lett. 105, 167006 (2010).

# Additional features
Due to their dependence on external libraries, certain TBTK features are not
installed by default. These are intended to eventually make it into the default
installation, but currently has to be configured manually to work. The current
components that do not work by default, and the corresponding reasons are
listed below. If you are interested in installing one or several of these
features already now, please contact the developer at
kristofer.bjornson@physics.uu.se for further instructions.

### Library
#### ArnoldiSolver
Depends on the external libraries ARPACK (http://www.caam.rice.edu/software/ARPACK/) and SuperLU v5.2.1 (http://crd-legacy.lbl.gov/~xiaoye/SuperLU/).

Given that ARPACK and SuperLU are installed, the ArnoldiSolver can be compiled by executing the following command from TBTK/Lib/.
```bash
make arnoldi
```
#### FourierTransform
Depends on the external library FFTW v3 (http://www.fftw.org/).

Given that FFTW is installed, the FourierTransform can be compiled by executing the following command from TBTK/Lib/.
```bash
make fourier
```

#### Plotter
Depends on the external library openCV (http://www.opencv.org/).

Given that openCV is installed, the Plotter can be compiled by executing the following command from TBTK/Lib/.
```bash
make plotter
```

#### RayTracer
Depends on the external library openCV (http://www.opencv.org/).

Given that openCV is installed, the RayTracer can be compiled by executing the following command from TBTK/Lib/.
```bash
make plotter
make raytracer
```

### Tools
#### TBTKImageToModel
Depends on the external library openCV (http://opencv.org/).

### Visualization
#### TBTKRayTracePlotter
Depends on the external library openCV (http://opencv.org/).

#### TBTKVisualizer
Depends on the external library OGRE (http://www.ogre3d.org/).

### Templates
#### BasicArnoldi
Depends on the library component ArnoldiSolver to have been compiled (see above).

#### RayTraceInteractive
Depends on the external library openCV (http://www.openCV.org/).

#### RayTracePlotter
Depends on the external library openCV (http://www.openCV.org/).

#### TopologicalInsulator3D
Depends on the external library FFTW (http://www.fftw.org/).
