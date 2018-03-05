Tutorials {#Tutorials}
======

These tutorials assume that TBTK has already been installed as described in the [Installation instructions](@ref InstallationInstructions).

# Getting started
- @subpage CreatingANewApplication

# Building applications
- @subpage BuildingAFirstApplication

@page CreatingANewApplication Creating a new application (Unix like operating systems such as Linux and Mac OS)
# Creating, building, and executing a first application {#CreatingBuildingAndExecutingAFirstApplication}
## Project creation using TBTKCreateApplication
One of the main issues when writing code in a language such as c++ is the need to manage both source files and dependencies in terms of other libraries.
In addition to the main source code, a full project therefore requires one or several supporting files and folders.
TBTK therefore comes with an executable called TBTKCreateApplication that helps setup the relevant support structure, and which in combination with CMake allows the developer to focus on developing the actuall application.
Before creating a project, it is recommended to have a folder called for example *TBTKApplications* in which different applications are created.
This can be created on the command line using
```bash
	mkdir TBTKApplications
```
Then, to enter this folder and create a new application, simply type
```bash
	cd TBTKApplications
	TBTKCreateApplication ApplicationName
```
where *ApplicationName* can be any valid folder name (no spaces).
This will create a new folder named *ApplicationName* that contains all the relevant files for building and running a TBTK application.
## Build executable
To enter the project folder and build the application, type
```bash
	cd ApplicationName
	cmake .
	make
```
Here the second line creates the relevant files required to build the application, while the third line builds the actuall executable.

It can also be useful to build the application in a separate folder from source code.
This can be achieved by instead typing
```bash
	mkdir ApplicationNameBuild
	cd ApplicationNameBuild
	cmake ../ApplicationName
	make
```

## Run the executable
The program can now be run by typing
```bash
	./build/Application
```

# Default application folder structure {#DefaultApplicationFolderStructure}
When creating an application project using TBTKCreateApplication, a number of files and folders are created.
The purpose of the folders are
| Folder name | Description                                |
|-------------|--------------------------------------------|
| build       | Contains the final executable application. |
| figures     | Intended for generated figures.            |
| include     | Contains header files (.h).                |
| src         | Contains source files (.cpp).              |

At creation the project also contains a number of files.
The files and their purpose are listed below.
Application developers can largly ignore files in gray text since these are meant to be unedited.
However, descriptions are provided to aid developers interested in customizing the build procedure.
| File name                                      | Description                                                                                                      |
|------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| <span style="color:gray">CMakeLists.txt</span> | <span style="color:gray">File used by CMake to setup the build system. The file used by the *cmake* call.</span> |
| plot.sh                                        | File intended for python plot commandin to be listed in (see TBTK/Templates for exampels).                       |
| src/main.cpp                                   | The file in which the code for the actuall application is written.                                               |

Once *cmake* is executed, a number of additional files are created
| File name | Description |
|-----------|-------------|
| <span style="color:gray">CMakeCache.txt</span>      | <span style="color:gray">See the [CMake](https://cmake.org/) documentation.</span> |
| <span style="color:gray">CMakeFiles (folder)</span> | <span style="color:gray">See the [CMake](https://cmake.org/) documentation.</span> |
| <span style="color:gray">cmake_install.cmake</span> | <span style="color:gray">See the [CMake](https://cmake.org/) documentation.</span> |
| <span style="color:gray">Makefile</span>            | <span style="color:gray">The file used by the *make* call.</span>                  |

# Template applications {#TemplateApplications}
## More complex templates
While the usage of TBTKCreateApplication as presented above is useful for starting projects from scratch, it is also possible to start from more complex projects.
```bash
	TBTKCreateApplication ApplicationName TemplateName
```
where *TemplateName* can be any of
| TemplateName                    | Description                                                                                                              |
|---------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| BasicArnoldi                    | Demonstrates how to use the ArnoldiIterator.                                                                             |
| BasicChebyshev                  | Demonstrates how to use the ChebyshevExapnder.                                                                           |
| BasicDiagonalization            | Demonstrates how to use the Diagonalizer.                                                                                |
| BasicFourierTransform           | Demonstrates how to use the FourierTransform.                                                                            |
| CarbonNanotube                  | Demonstrates how to set up a carbon nanotube.                                                                            |
| HexagonalLattice                | Demonstrates how to set up a hexagonal lattice.                                                                          |
| PartialBilayer                  | Demonstrates how to set up a partial bilayer.                                                                            |
| SelfConsistentSuperconductivity | Demonstrates how to set up a self-consistent calculation for the superconducting order parameter using the Diagonalizer. |
| WireOnSuperconductor            | Demonstrates how to set up a magnetic wire on top of a two-dimensional superconducting layer.                            |

## Example: BasicDiagonalization
To demonstrate this, lets build and execute BasicDiagonalization.
Before beginning we note that this template uses the FileWriter and therefore requires that HDF5 is installed and detected by TBTK.
Therefore make sure the HDF5 box is checked in the output from *cmake* below.
Starting from the folder TBTKApplication, we type
```bash
	TBTKCreateApplication MyDiagonalizationApplication BasicDiagonalization
	cd MyDiagonalizationApplication
	cmake .
	make
	./build/Application
```
The application should run and output
```bash
	Constructing system
		Basis size: 800
	Initializing Diagonalizer
		Basis size: 800
	Running Diagonalizer

	.
```
Taking a look at the code in *src/main.cpp*, we see several lines for example reading
```cpp
	FileWriter::writeDensity(density);
```
These lines writes the results to a file called *TBTKResults.h5* (set by *FileWriter::setFileName("TBTKResults.h5")*) which will contain the results of the calculation.
Next, taking a look in the file *plot.sh*, we see several corresponding lines for example reading
```bash
	TBTKPlotDensity.py TBTKResults.h5
```
These reads the results from *TBTKResults.h5* and plots the result in files stored in the folder *figures*.
Therefore type
```bash
	bash plot.sh
```
and view the results in the *figures* folder.

@page BuildingAFirstApplication Building a first application
# Description
As a first example we will create an application that models a simple two level system.
```bash
	TBTKCreateApplication TwoLevelSystem
```
