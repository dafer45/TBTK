Installation instructions {#InstallationInstructions}
======

# System requirements
In order to install TBTK, the following libraries must be installed on the system
| Required libraries | Further information           |
|--------------------|-------------------------------|
| BLAS               | http://www.netlib.org/blas/   |
| LAPACK             | http://www.netlib.org/lapack/ |

# Downloading TBTK {#DownloadingTBTK}
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

# Installation {#Installation}
There are currently two ways to install TBTK.
One traditional way using makefiles and installation scripts and one based on CMake (https://cmake.org).
The intention is to fully transition to CMake since the first method only is compatible with Unix based systems such as Linux and Mac OS.
However, it is currently recommended that application developers use the traditional method since CMake support still is under development.

## Traditional installation
### Initialize session
Whenever a new session is started (a new terminal window is opened), the first thing that needs to be done is to initialize a current TBTK session.
This is true both before installation and every time the library is used after installation.
This is done by entering the TBTK folder and typing
```bash
	source init_session.sh
```

### Install
Some calculations in TBTK can be carried out on CUDA compatible GPUs.
Depending on whether the computer has a CUDA compatible graphics card or not, type one of the two following commands.

With CUDA support:
```bash
	bash install.sh -CUDA
```
Without CUDA support
```bash
	bash install.sh
```

### Update library
If TBTK already is installed on the system, it is possible to update to the latest version by typing
```bash
	git pull
	git checkout VERSION_TAG
```
Once the latset version of the code has been pulled, the library can be reinstalled as follows.

With CUDA support
```bash
	bash update.sh -CUDA
```
Without CUDA support
```bash
	bash update.sh
```

### Install additional features
One of the main drawbacks of the traditional method is that it does not allow for automatic detection of external libraries that allow TBTK to extend its core functionalities.
These libraries are
| Optional libraries | Further information                        |
|--------------------|--------------------------------------------|
| ARPACK             | http://www.caam.rice.edu/software/ARPACK/  |
| FFTW3              | http://www.fftw.org/                       |
| OpenCV             | https://opencv.org/                        |
| cURL               | https://curl.haxx.se/                      |
| SuperLU (v5.2.1)   | http://crd-legacy.lbl.gov/~xiaoye/SuperLU/ |

If one or several of these libraries are installed, it is possible to enter the TBTK/Lib folder and install further components by typing the following commands.

Requires ARPACK and SuperLU
```bash
	make arnoldi
	make lu
```

Requires FFTW3
```bash
	make fourier
```

Requires OpenCV
```bash
	make plotter
```

Requires OpenCV
```bash
	make raytracer
```

Requires cURL
```bash
	make resource
```

## Installation using CMake
To install using CMake create a new folder, which we here call TBTKBuild.
Enter this folder and type
```bash
	cmake /path/to/TBTK
	make
	make install
```

