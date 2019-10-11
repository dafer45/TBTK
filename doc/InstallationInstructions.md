Installation instructions {#InstallationInstructions}
======

# System requirements {#SystemRequirements}
| Required software and libraries | Further information          |
|---------------------------------|------------------------------|
| git                             | https://git-scm.com/         |
| CMake (Minimum version 3.0)     | https://cmake.org            |
| BLAS                            | http://www.netlib.org/blas   |
| LAPACK                          | http://www.netlib.org/lapack |

# Download TBTK {#DownloadTBTK}

```cpp
	git clone https://github.com/dafer45/TBTK/
```

# Select version
Always work agains a public release.
This is done by checking out the latset release using
```cpp
	git checkout v1.1.2
```
A full list of public releases can be found at https://github.com/dafer45/TBTK/releases.
By remembering the version number that an application is compiled with, it is guaranted that the application will be possible to build again anytime in the future.

TBTK uses [semantic versioning](https://semver.org/spec/v2.0.0.html).
This means that any application that compiles with v1.x.y also will work with any higher version v1.x'.y'.
An update to the first (major) version number may break older applications.

# Install TBTK {#InstallTBTK}
## Unix like operating systems such as Linux and Mac OS
Begin by creating a build folder in parallel with the TBTK source folder (i.e. not inside).
```cpp
	mkdir TBTKBuild
```
The folder structure should now be
```cpp
	SomeFolder/
		TBTK
		TBTKBuild
```
Then do the following.
```cpp
	cd TBTKBuild
	cmake ../TBTK
	make
	sudo make install
```

## Local installation
It is possible to perform a local install if administrator privileges are not available.
To do so, add the flag -DCMAKE_INSTALL_PREFIX=install/path/ to the *cmake* call above, where install/path/ is the preferred installation path.
Make sure to also set the environment variables CPLUS_INCLUDE_PATH, LIBRARY_PATH, and PATH if this is done.
The paths should be install/path/include/, install/path/lib/, and install/path/bin/, respectively.

## Customized installation
It is possible to customize the build by passing these flags to cmake.
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

The following table shows the optional libraries that are required to enable the respective TBTK components.
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

@link Manual Next: Manual@endlink
