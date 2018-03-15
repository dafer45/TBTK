Installation instructions {#InstallationInstructions}
======

# System requirements {#SystemRequirements}
In order to install TBTK, the following software must be installed on the system.
| Required software | Further information  |
|-------------------|----------------------|
| git               | https://git-scm.com/ |
| CMake             | https://cmake.org    |

The following libraries are also required.
If you are unsure whether the libraries are installed or not, you can proceed with the installation as described below.
If these libraries are missing, the call to *cmake* will fail with an error message that tells that one or both of these are missing.
| Required libraries | Further information               |
|--------------------|-----------------------------------|
| BLAS               | http://www.netlib.org/blas        |
| LAPACK             | http://www.netlib.org/lapack      |


## Optional requirements
Additional features will also be available if one or more of the following libraries are installed.
| Optional libraries | Further information                         |
|--------------------|---------------------------------------------|
| ARPACK             | http://www.caam.rice.edu/software/ARPACK    |
| FFTW3              | http://www.fftw.org                         |
| OpenCV             | https://opencv.org                          |
| cURL               | https://curl.haxx.se                        |
| SuperLU (v5.2.1)   | http://crd-legacy.lbl.gov/~xiaoye/SuperLU   |
| wxWidgets          | https://www.wxwidgets.org                   |
| CUDA               | https://developer.nvidia.com/cuda-downloads |
| HDF5               | https://support.hdfgroup.org/HDF5           |

The following table shows the optional libraries that are required for the different TBTK components.
|                                            | ARPACK | FFTW3 | OpenCV | cURL | SuperLU (v5.2.1) | wxWidgets | CUDA | HDF5 |
|--------------------------------------------|:------:|:-----:|:------:|:----:|:----------------:|:---------:|:----:|:----:|
| ArnoldiIterator                            | X      |       |        |      | X                |           |      |      |
| FourierTransform                           |        | X     |        |      |                  |           |      |      |
| Plotter                                    |        |       | X      |      |                  |           |      |      |
| RayTracer                                  |        |       | X      |      |                  |           |      |      |
| Resource                                   |        |       |        | X    |                  |           |      |      |
| DataManager                                |        |       |        | X    |                  |           |      |      |
| LinnearEquationSolver                      |        |       |        |      | X                |           |      |      |
| LUSolver                                   |        |       |        |      | X                |           |      |      |
| GUI                                        |        |       |        |      |                  | X         |      |      |
| Enable GPU execution for ChebyshevExpander |        |       |        |      |                  |           | X    |      |
| FileReader and FileWriter                  |        |       |        |      |                  |           |      | X    |

# Download TBTK {#DownloadTBTK}
TBTK can be downloaded from github, which is done by typing
```bash
	git clone https://github.com/dafer45/TBTK/
```

[//]: # # Select version
[//]: # TBTK is still in a phase where changes to the core API may occur, even if most core components are relatively stable by now.
[//]: # To be able to know that a particular application compiles also in the future, it is therefore recommended that application developers work against one of the public releases.
[//]: # A list of releases and their name tag can be found on https://github.com/dafer45/TBTK/releases.
[//]: # To use a particular version, for example v0.9.5, stand in the TBTK root folder and type
[//]: # ```bash
	git checkout v0.9.5
[//]: # ```
[//]: # It is recommended that application developers stores a note insde any project using the library which tells against which version of the library it has been compiled.
[//]: # It is then possible at any time in the future to check out v0.9.5 and recompile an application developed against this version.

# Install TBTK {#InstallTBTK}
## Unix like operating systems such as Linux and Mac OS
TBTK should be built in a different folder than the source folder, therefore begin by create a new folder outside of the TBTK folder, for example TBTKBuild.
Then enter this folder and type
```bash
	cmake /path/to/TBTK
```

Compile the library by typing
```bash
	make
```

Finally, TBTK is installed by typing
```bash
	sudo make install
```
The last command requires administrator privileges.
If administrator privileges are not available, or a local install is preferred, see the customization options below.

## Customized installation
The installation can be customized by supplying the following flags to the cmake command above
| Flag                             | Description                                                                                                |
|----------------------------------|------------------------------------------------------------------------------------------------------------|
| -DCMAKE_INSTALL_PREFIX:PATH=XXX  | Specify a custom insallation path. For example a local path if administrator privileges are not available. |
| -DSUPER_LU_INCLUDE_PATH:PATH=XXX | Specify a non standard search path for SuperLU include files.                                              |
| -DSUPER_LU_LIBRARY_PATH:PATH=XXX | Specify a non standard search path for SuperLU library files.                                              |

where XXX is to be replaced by the relevant path.
