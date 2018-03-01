Installation instructions {#InstallationInstructions}
======

# System requirements {#SystemRequirements}
In order to install TBTK, the following software and libraries must be installed on the system
| Required software | Further information |
|-------------------|---------------------|
| CMake             | https://cmake.org   |

| Required libraries | Further information               |
|--------------------|-----------------------------------|
| BLAS               | http://www.netlib.org/blas        |
| LAPACK             | http://www.netlib.org/lapack      |
| HDF5               | https://support.hdfgroup.org/HDF5 |

## Optional requirements
Additional features will also be available if one or more of the following libraries are installed
| Optional libraries | Further information                         |
|--------------------|---------------------------------------------|
| ARPACK             | http://www.caam.rice.edu/software/ARPACK    |
| FFTW3              | http://www.fftw.org                         |
| OpenCV             | https://opencv.org                          |
| cURL               | https://curl.haxx.se                        |
| SuperLU (v5.2.1)   | http://crd-legacy.lbl.gov/~xiaoye/SuperLU   |
| wxWidgets          | https://www.wxwidgets.org                   |
| CUDA               | https://developer.nvidia.com/cuda-downloads |

The following table shows the optional libraries that are required for the different TBTK components
|                                            | ARPACK | FFTW3 | OpenCV | cURL | SuperLU (v5.2.1) | wxWidgets | CUDA |
|--------------------------------------------|:------:|:-----:|:------:|:----:|:----------------:|:---------:|:----:|
| ArnoldiIterator                            | X      |       |        |      | X                |           |      |
| FourierTransform                           |        | X     |        |      |                  |           |      |
| Plotter                                    |        |       | X      |      |                  |           |      |
| RayTracer                                  |        |       | X      |      |                  |           |      |
| Resource                                   |        |       |        | X    |                  |           |      |
| DataManager                                |        |       |        | X    |                  |           |      |
| LinnearEquationSolver                      |        |       |        |      | X                |           |      |
| LUSolver                                   |        |       |        |      | X                |           |      |
| GUI                                        |        |       |        |      |                  | X         |      |
| Enable GPU execution for ChebyshevExpander |        |       |        |      |                  |           | X    |

# Download TBTK {#DownloadTBTK}
TBTK can be downloaded from github.
Assuming git (https://git-scm.com/) is installed, type
```bash
	git clone https://github.com/dafer45/TBTK/
```

# Select version
TBTK is still in a phase where changes to the core API may occur, even if most core components are relatively stable by now.
To be able to know that a particular application compiles also in the future, it is therefore recommended that application developers work against one of the public releases.
A list of releases and their name tag can be found on https://github.com/dafer45/TBTK/releases.
To use a particular version, for example v0.9.5, stand in the TBTK root folder and type
```bash
	git checkout v0.9.5
```
It is recommended that application developers stores a note insde any project using the library which tells against which version of the library it has been compiled.
It is then possible at any time in the future to check out v0.9.5 and recompile an application developed against this version.

# Install TBTK {#InstallTBTK}
Begin by create a new folder, then enter this folder and type
```bash
	cmake /path/to/TBTK
```
The installation can be customized by supplying the following flags to the cmake command above
| Flag                            | Description                                                   |
|---------------------------------|---------------------------------------------------------------|
| -DCMAKE_INSTALL_PREFIX:PATH=XXX | Specify a custom insallation path.                            |
| -DSUPER_LU_INCLUDE_PATH=XXX     | Specify a non standard search path for SuperLU include files. |
| -DSUPER_LU_LIBRARY_PATH=XXX     | Specify a non standard search path for SuperLU library files. |

where XXX is to be replaced by the relevant path.

Then compile the library by typing
```bash
	make
```

Finally, TBTK is installed by typing
```bash
	make install
```
