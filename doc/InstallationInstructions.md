Installation instructions {#InstallationInstructions}
======

# System requirements {#SystemRequirements}
In order to install TBTK, the following software must be installed.
| Required software           | Further information  |
|-----------------------------|----------------------|
| git                         | https://git-scm.com/ |
| CMake (Minimum version 3.0) | https://cmake.org    |

The following libraries are also required.
If you are unsure whether the libraries are installed or not, you can proceed with the installation as described below.
If a required library is missing, the call to *cmake* will fail with an error message indicating the missing library.
| Required libraries | Further information               |
|--------------------|-----------------------------------|
| BLAS               | http://www.netlib.org/blas        |
| LAPACK             | http://www.netlib.org/lapack      |

There are also optional libraries that if installed will enable further components of TBTK.
For a list of these libraries and the TBTK components that are enabled if they are installed, see the Extensions section below.

# Download TBTK {#DownloadTBTK}
TBTK can be downloaded from github, which is done by typing
```bash
	git clone https://github.com/dafer45/TBTK/
```

# Select version
TBTK is still in a phase where changes to the core API may occur, even if most core components are relatively stable by now.
To be able to know that a particular application compiles also in the future, it is recommended that application developers work against one of the public releases.
A list of releases and their name tags can be found on https://github.com/dafer45/TBTK/releases.
To use a particular version, for example v1.1.1, execute the following from the TBTK source folder
```bash
	git checkout v1.1.1
```
It is recommended to store a note inside projects using TBTK that tells against which version of the library it has been compiled.
This makes it possible to anytime in the future checkout the exact same version of TBTK and recompile an application developed against it.
If a named version such as v1.1.1 is not used, the developer should instead remember the exact git hash.

# Install TBTK {#InstallTBTK}
## Unix like operating systems such as Linux and Mac OS
TBTK should be built in a different folder than the source folder, therefore begin by creating a new folder outside of the TBTK folder, for example TBTKBuild.
Then enter this folder and type
```bash
	cmake /path/to/TBTK
```
where '/path/to/TBTK' is to be replaced by the actual path to the TBTK source folder.

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

# Extensions {#Extensions}
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

