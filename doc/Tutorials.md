Tutorials {#Tutorials}
======

These tutorials assume that TBTK has already been installed as described in the [Installation instructions](@ref InstallationInstructions).
The tutorials are written for physicists that are not necessarily familiar with C++ and therefore introduces both TBTK and C++.
See the Examples for quicker information about how to implement specific problems.

# Getting started
- @subpage CreatingANewApplication

# Building applications
- @subpage BuildingAFirstApplicationTwoLevelSystem

@page CreatingANewApplication Creating a new application (Unix like operating systems such as Linux and Mac OS)
# Purpose and learning outcome {#CreatingANewApplicationPurposeAndLearningOutcome}
In this tutorial we learn how to create, build and execute an application.
We also describe the default folder structure and learn how to create template projects.
At the end of this tutorial you should be comfortable with setting up a new project and be ready to learn how to implement an applications.

# Creating, building, and executing a first application {#CreatingBuildingAndExecutingAFirstApplication}
## Create an application
We begin by creating a folder within which applications will be created.
To create this folder and enter it, we type the following on the command line.
```txt
	mkdir TBTKApplications
	cd TBTKApplications
```

Once inside the folder we type
```txt
	TBTKCreateApplication ApplicationName
```
where *ApplicationName* can be any valid folder name (without spaces).
This creates a new folder named *ApplicationName* that contains all the relevant files for building and running a TBTK application.

## Build executable
To enter the project folder and build the application, type
```txt
	cd ApplicationName
	cmake .
	make
```
Here the second line creates the relevant files required to build the application, while the third line builds the actual executable.

It can also be useful to build the application in a separate folder that is separate from the source code.
This can be achieved by instead typing
```txt
	mkdir ApplicationNameBuild
	cd ApplicationNameBuild
	cmake ../ApplicationName
	make
```

## Execute the application
The application can now be executed by typing
```txt
	./build/Application
```
That's it, you have now created, built, and executed your first application.

# Default application folder structure {#DefaultApplicationFolderStructure}
When running TBTKCreateApplication, a number of files and folders are created inside the new application folder.
The purpose of the folders are
| Folder name | Description                                |
|-------------|--------------------------------------------|
| build       | Contains the final executable application. |
| figures     | Intended for generated figures.            |
| include     | Contains header files (.h).                |
| src         | Contains source files (.cpp).              |

At creation the project also contains a number of files.
The files and their purpose are listed below.
Application developers can largely ignore files in gray text since these are meant to be unedited.
However, descriptions are provided to aid developers interested in customizing the build procedure.
| File name                                      | Description                                                                                                      |
|------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| <span style="color:gray">CMakeLists.txt</span> | <span style="color:gray">File used by CMake to setup the build system. The file used by the *cmake* call.</span> |
| src/main.cpp                                   | The file in which the code for the actual application is written.                                                |

Once *cmake* is executed, a number of additional files are created
| File name | Description |
|-----------|-------------|
| <span style="color:gray">CMakeCache.txt</span>      | <span style="color:gray">See the [CMake](https://cmake.org/) documentation.</span> |
| <span style="color:gray">CMakeFiles (folder)</span> | <span style="color:gray">See the [CMake](https://cmake.org/) documentation.</span> |
| <span style="color:gray">cmake_install.cmake</span> | <span style="color:gray">See the [CMake](https://cmake.org/) documentation.</span> |
| <span style="color:gray">Makefile</span>            | <span style="color:gray">The file used by the *make* call.</span>                  |

# Template applications {#TemplateApplications}
## More complex templates
So far we have shown how to create an application from scratch using TBTKCreateApplication.
It is also possible to create one of several template projects by instead typing
```txt
	TBTKCreateApplication ApplicationName TemplateName
```
where *TemplateName* can be any of the following
| TemplateName                    | Description                                                                                                              |
|---------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| BasicArnoldi                    | Demonstrates how to use the Solver::ArnoldiIterator.                                                                     |
| BasicChebyshev                  | Demonstrates how to use the Solver::ChebyshevExapnder.                                                                   |
| BasicDiagonalization            | Demonstrates how to use the Solver::Diagonalizer.                                                                        |
| BasicFourierTransform           | Demonstrates how to use the FourierTransform.                                                                            |
| BasicLinearEquation             | Demonstrates how to use the Solver::BasicLinearEquation.                                                                 |
| CarbonNanotube                  | Demonstrates how to set up a carbon nanotube.                                                                            |
| HexagonalLattice                | Demonstrates how to set up a hexagonal lattice.                                                                          |
| PartialBilayer                  | Demonstrates how to set up a partial bilayer.                                                                            |
| SelfConsistentSuperconductivity | Demonstrates how to set up a self-consistent calculation for the superconducting order parameter using the Diagonalizer. |
| WireOnSuperconductor            | Demonstrates how to set up a magnetic wire on top of a two-dimensional superconducting layer.                            |

## Example: BasicDiagonalization
To demonstrate the use of template projects, let's build and execute BasicDiagonalization.
We note that this template uses the Plotter and, therefore, requires that Python with matplotlib and numpy is installed and detected by TBTK.
Therefore, make sure that the Python box is checked in the output from the *cmake* command below before proceeding to line four and five.

Starting from the folder TBTKApplication, we type
```txt
	TBTKCreateApplication MyDiagonalizationApplication BasicDiagonalization
	cd MyDiagonalizationApplication
	cmake .
	make
	./build/Application
```
The application should run and save six figures in the figures folder.

@page BuildingAFirstApplicationTwoLevelSystem Building a first application (two level system)
# Purpose and learning outcome {#BuildingAFirstApplicationTwoLevelSystemPurposeAndLearningOutcome}
In this tutorial we create an application that models a simple two level system.
The problem itself is very simple and is more easily solved using pen and paper.
However, this simple problem allows us to focus on getting familiar with both C++ and the general workflow for writing TBTK applications.

At the end of this tutorial the you will have a good understanding of the general structure of a TBTK application and be able to start writing custom applications.
You will know how to use the UnitHandler to set up the natural units for the application.
You will also know how to set up a Model, choose a Solver, and use a PropertyExtractor to calculate properties of interest.

# Problem description {#ProblemDescription}
Consider a single electron spin in a magnetic field described by the Hamiltonian
<center>\f$ H = -\mu_B B\sigma_z\f$,</center>
where \f$\mu_B\f$ is the Bohr magneton, B is the magnetic field strength, and \f$\sigma_z\f$ is the third Pauli matrix.
The energy split induced by a magnetic field acting on an electron spin is known as the Zeeman split.
In this tutorial we write an application that can be used to calculate this energy split as a function of the magnetic field.
Further, at zero temperature and non-zero magnetic field strength, the system will be in its ground state and be maximally magnetized parallel to the magnetic field.
However, at finite temperatures both the ground state and the excited state will have a finite occupation determined by the Fermi-Dirac distribution.
We therefore also invstigate how the magnetization varies with magnetic field strength and temperature.

# Rewriting the Hamiltonian on second quantized form {#RewritingTheHamiltonianOnSecondQuantizedForm}
Before implementing the problem we need to rewrite the Hamiltonian on the second quantized form
<center>\f$H = \sum_{\mathbf{i}\mathbf{j}}a_{\mathbf{i}\mathbf{j}}c_{\mathbf{i}}^{\dagger}c_{\mathbf{j}}\f$.</center>
For the Hamiltonian above this is particularly simple.
First of all the indices \f$\mathbf{i}\f$ and \f$\mathbf{j}\f$ are simply \f$\uparrow\f$ and \f$\downarrow\f$ for up and down spins, respectively.
Second, the Pauli matrix \f$\sigma_z\f$ only have the non-zero entries \f$1\f$ and \f$-1\f$ on the diagonal.
The Hamiltonian can therefore be written as
<center>\f$H = -\mu_B Bc_{\uparrow}^{\dagger}c_{\uparrow} + \mu_B Bc_{\downarrow}^{\dagger}c_{\downarrow}\f$.</center>
Identifying up spin with \f$0\f$ and down spin with \f$1\f$, the hopping amplitudes \f$a_{\mathbf{i}\mathbf{j}}\f$ can then be tabulated as
| Hopping amplitude              | Value          | To Index | From Index |
|--------------------------------|----------------|----------|------------|
| \f$a_{\uparrow\uparrow}\f$     | \f$-\mu_B B\f$ | {0}      | {0}        |
| \f$a_{\downarrow\downarrow}\f$ | \f$\mu_B B\f$  | {1}      | {1}        |
Here the first column is an analytical symbol for the hopping amplitude, the second is its actual value, while the third and forth column corresponds to the numerical representation of the first and second index of \f$a_{\mathbf{i}\mathbf{j}}\f$.
The three last columns forms a complete representation of the hopping amplitude and is what enters into the numerical model.
This way of representing the hopping amplitudes reoccurs throughout TBTK and the documentation.
See [Model: A more complex example](@ref AMoreComplexExample) for more information.

# Implementation {#Implementation}
## Understanding *src/main.cpp* (C++ crash course)
We are now ready to start the implementation and we here assume that an empty project has been created as described in the tutorial [Creating a new application](@ref CreatingANewApplication).
If we open the file *src/main.cpp*, we see the following code
```cpp
	#include "TBTK/Streams.h"
	#include "TBTK/TBTK.h"

	#include <complex>

	using namespace std;
	using namespace TBTK;

	int main(int argc, char **argv){
		//Initialize TBTK.
		Initialize();

		Streams::out << "Hello quantum world!\n";

		return 0;
	}
```
Here the first three lines are include statements that includes other C++ library components in the code.
The first include statement includes a component called *Streams* from TBTK that can be used to output information to the screen at runtime.
The second line makes TBTK's initialization function available.
Finally, the third include statement includes *complex* numbers from the C++ standard library (STL).
In the empty project complex number are actually not used, but since complex numbers are so common in TBTK applications they are included by default in the template projects.
In general, each TBTK component is made available by including a separate file and as we build our application we will be including more files.

To understand the next two lines we note that in C++, functions, classes, etc. can be organized into namespaces to avoid name clashes between different libraries.
In particular, every component in STL is in the namespace *std*, while every component of TBTK is in the namespace *TBTK*.
For example, the Streams class included above is by default accessed as TBTK::Streams.
However, often there are few actual name clashes and it is convenient to not have to prepend every function call etc. with *std::* or *TBTK::* and it is therefore possible to tell the compiler to "use" particular namespaces.
Possible name clashes can then be handled individually by prepending only those components for which ambiguity arise by their corresponding namespaces.

The actual entry point for the program is
```cpp
	int main(int argc, char **argv){
```
However, we note that global variable assignments occur ahead of this time, meaning that it in principle is possible for some execution of code to occur already before the main routine begins.
The arguments *argc* and *argv* can be used to accept arguments from the command line, and the interested reader is advised to make a web search for further information on how to use these.

Inside the main function three things occur.
First, TBTK is initialized with the call to *Initialize()*.
Second, "Hello quantum world!\n" is written to the output stream, where '<<' can be read as send the right hand side to the output stream Streams::out.
Writing to Streams::out will by default result in information being printed to the screen.
The "\n" at the end of the string is a line break character and means that any character that is printed after it will appear on a new line.
Finally, the main function exits by returning *0*, which is a message indicating that the application finished as expected.

## Initializing units
Having understood the structure of *src/main.cpp*, we are now ready to implement the actual application.
The first step is to specify the units that we will be using.
Of particular interest to us is to measure temperature in terms of Kelvin and energies in terms of *meV*.
The remaining quantities we set to rad, Coulomb, pieces, meter, and seconds.
To achieve this we remove the print statement in the code above and replace it by
```cpp
	#include "TBTK/TBTK.h"
	#include "TBTK/UnitHandler.h"

	#include <complex>

	using namespace std;
	using namespace TBTK;

	int main(int argc, char **argv){
		//Initialize TBTK.
		Initialize();

		//Initialize the UnitHandler.
		UnitHandler::setScales(
			{"1 rad", "1 C", "1 pcs", "1 meV", "1 m", "1 K", "1 s"}
		);

		return 0;
	}
```
All numbers passed to TBTK functions will be interpreted to be in these units.
This, for example, means that a 2 passed to a function that accepts an energy will be interpreted as 2 meV.

## Specifying parameters
Our next step is to set up variables containing the relevant parameters for the problem.
We begin by specifying some default parameter values *T = 300 K* and *B = 1 T*, as well as requesting the Bohr magneton from the UnitHandler.
```cpp
	#include "TBTK/TBTK.h"
	#include "TBTK/UnitHandler.h"

	#include <complex>

	using namespace std;
	using namespace TBTK;

	int main(int argc, char **argv){
		//Initialize TBTK.
		Initialize();

		//Initialize the UnitHandler.
		UnitHandler::setScales(
			{"1 rad", "1 C", "1 pcs", "1 meV", "1 m", "1 K", "1 s"}
		);

		//Set up input parameters.
		double T = 300;
		double B = UnitHandler::convertArbitraryToNatural<
			Quantity::MagneticField
		>(
			1,
			Quantity::MagneticField::Unit::T
		);

		//Get the Bohr magneton.
		double mu_B = UnitHandler::getConstantInNaturalUnits("mu_B");

		return 0;
	}
```
Since the natural units are set to Kelvin for temperatures, we can specify *T* to be *300*.
However, the unit Tesla for magnetic fields is not among any of the seven units specified in the UnitHandler.
In fact, magnetic field is not a [base quantity](@ref BaseUnits) but a [derived quantity](@ref DerivedUnits) and care therefore have to be taken when specifying the value.
The function *UnitHandler::convertArbitraryToNatural()* is therefore used to convert the magnetic field from Tesla to the corresponding natural unit implied by the choice of units passed to the *UnitHandler*.
In the last new line the Bohr magneton is requested in natural units.

## Setting up the model
Having specified the parameters for the problem, the Model can be setup as follows.
```cpp
	#include "TBTK/Model.h"
	#include "TBTK/TBTK.h"
	#include "TBTK/UnitHandler.h"

	#include <complex>

	using namespace std;
	using namespace TBTK;

	int main(int argc, char **argv){
		//Initialize TBTK.
		Initialize();

		//Initialize the UnitHandler.
		UnitHandler::setScales(
			{"1 rad", "1 C", "1 pcs", "1 meV", "1 m", "1 K", "1 s"}
		);

		//Setup input parameters.
		double T = 300;
		double B =  UnitHandler::convertArbitraryToNatural<
			Quantity::MagneticField
		>(
			1,
			Quantity::MagneticField::Unit::T
		);

		//Get Bohr magneton.
		double mu_B = UnitHandler::getConstantInNaturalUnits("mu_B");

		//Create model object.
		Model model;

		//Add HoppingAmplitudes to the Model.
		model << HoppingAmplitude(-mu_B*B, {0}, {0});
		model << HoppingAmplitude( mu_B*B, {1}, {1});

		//Construct a Hilbert space for the Model.
		model.construct();

		//Set the temperature.
		model.setTemperature(T);

		return 0;
	}
```
Here the first new line creates the actual Model object called *model* and in the two following lines the HoppingAmplitudes are added to the Model.
Note that the syntax for feeding HoppingAmplitudes to the Model is similar to how the string "Hello quantum world!\n" was fed to Streams::out.
Also note how the HoppingAmplitudes are constructed with the arguments in the same order as the three last columns in the hopping amplitude table at the beginning of this tutorial.
Next, the second last new line sets up the actual Hilbert space for the problem, which is necessary to do before passing it on to the Solver.
Finally, the temperature is set in the last new line.

## Choosing Solver
For a small problem like the one considered here the best solution method often is diagonalization, which after solving the problem gives direct access to the eigenvalues and eigenstates.
We therefore here setup and run a Solver::Diagonalizer as follows.
```cpp
	#include "TBTK/Model.h"
	#include "TBTK/Solver/Diagonalizer.h"
	#include "TBTK/TBTK.h"
	#include "TBTK/UnitHandler.h"

	#include <complex>

	using namespace std;
	using namespace TBTK;

	int main(int argc, char **argv){
		//Initialize TBTK.
		Initialize();

		//Initialize the UnitHandler.
		UnitHandler::setScales(
			{"1 rad", "1 C", "1 pcs", "1 meV", "1 m", "1 K", "1 s"}
		);

		//Setup input parameters.
		double T = 300;
		double B =  UnitHandler::convertArbitraryToNatural<
			Quantity::MagneticField
		>(
			1,
			Quantity::MagneticField::Unit::T
		);

		//Get the Bohr magneton.
		double mu_B = UnitHandler::getConstantInNaturalUnits("mu_B");

		//Create model object.
		Model model;

		//Add HoppingAmplitudes to the Model.
		model << HoppingAmplitude(-mu_B*B, {0}, {0});
		model << HoppingAmplitude( mu_B*B, {1}, {1});

		//Construct a Hilbert space for the Model.
		model.construct();

		//Set the temperature.
		model.setTemperature(T);

		//Create Solver.
		Solver::Diagonalizer solver;
		solver.setModel(model);
		solver.run();

		return 0;
	}
```
In the second new line the Solver is told to work on the *model*, while the last new line runs the diagonalization procedure.

## Creating a PropertyExtractor
Because different Solvers can present themselves very differently to the outside world, direct extraction of properties from the Solver is discouraged in TBTK.
Instead, Solvers come with corresponding PropertyExtractors that abstracts away some of the irrelevant numerical details of the particular Solvers and allows for focus to be put on the actual physics of the problem.
The next step is therefore to wrap the Solver in a PropertyExtractor, which is done as follows.
```cpp
	#include "TBTK/Model.h"
	#include "TBTK/Solver/Diagonalizer.h"
	#include "TBTK/TBTK.h"
	#include "TBTK/UnitHandler.h"
	#include "TBTK/PropertyExtractor/Diagonalizer.h"

	#include <complex>

	using namespace std;
	using namespace TBTK;

	int main(int argc, char **argv){
		//Initialize TBTK.
		Initialize();

		//Initialize the UnitHandler.
		UnitHandler::setScales(
			{"1 rad", "1 C", "1 pcs", "1 meV", "1 m", "1 K", "1 s"}
		);

		//Setup input parameters.
		double T = 300;
		double B =  UnitHandler::convertArbitraryToNatural<
			Quantity::MagneticField
		>(
			1,
			Quantity::MagneticField::Unit::T
		);

		//Get the Bohr magneton.
		double mu_B = UnitHandler::getConstantInNaturalUnits("mu_B");

		//Create model object.
		Model model;

		//Add HoppingAmplitudes to the Model.
		model << HoppingAmplitude(-mu_B*B, {0}, {0});
		model << HoppingAmplitude( mu_B*B, {1}, {1});

		//Construct a Hilbert space for the Model.
		model.construct();

		//Set the temperature.
		model.setTemperature(T);

		//Create Solver.
		Solver::Diagonalizer solver;
		solver.setModel(model);
		solver.run();

		//Create PropertyExtractor.
		PropertyExtractor::Diagonalizer propertyExtractor(solver);

		return 0;
	}
```

## Calculating the Zeeman split
To calculate the Zeeman split, we need to know the energy of the individual eigenstates of the system.
In our case we have two eigenstates with eigenvalues \f$E_0\f$ and \f$E_1\f$ and the Zeeman split is calculated as \f$E_Z = E_1 - E_0\f$.
We implement these calculations as follows.
```cpp
	#include "TBTK/Model.h"
	#include "TBTK/PropertyExtractor/Diagonalizer.h"
	#include "TBTK/Property/EigenValues.h"
	#include "TBTK/Solver/Diagonalizer.h"
	#include "TBTK/Streams.h"
	#include "TBTK/UnitHandler.h"

	#include <complex>

	using namespace std;
	using namespace TBTK;

	int main(int argc, char **argv){
		//Initialize TBTK.
		Initialize();

		//Initialize the UnitHandler.
		UnitHandler::setScales(
			{"1 rad", "1 C", "1 pcs", "1 meV", "1 m", "1 K", "1 s"}
		);

		//Setup input parameters.
		double T = 300;
		double B =  UnitHandler::convertArbitraryToNatural<
			Quantity::MagneticField
		>(
			1,
			Quantity::MagneticField::Unit::T
		);

		//Get the Bohr magneton.
		double mu_B = UnitHandler::getConstantInNaturalUnits("mu_B");

		//Create model object.
		Model model;

		//Add HoppingAmplitudes to the Model.
		model << HoppingAmplitude(-mu_B*B, {0}, {0});
		model << HoppingAmplitude( mu_B*B, {1}, {1});

		//Construct a Hilbert space for the Model.
		model.construct();

		//Set the temperature.
		model.setTemperature(T);

		//Create Solver.
		Solver::Diagonalizer solver;
		solver.setModel(model);
		solver.run();

		//Create PropertyExtractor.
		PropertyExtractor::Diagonalizer propertyExtractor(solver);

		//Calculate eigenvalues
		Property::EigenValues eigenValues
			= propertyExtractor.getEigenValues();

		//Print the energies of the individual eigenstates.
		Streams::out << "Energies for the individual eigenstates:\n";
		for(unsigned int n = 0; n < eigenValues.getSize(); n++){
			Streams::out << UnitHandler::convertNaturalToBase<
				Quantity::Energy
			>(
				eigenValues(n)
			) << " "
			<< UnitHandler::getUnitString<Quantity::Energy>()
			<< "\n";
		}

		//Print the Zeeman split.
		Streams::out << "\nZeeman split: "
			<< UnitHandler::convertNaturalToBase<Quantity::Energy>(
				eigenValues(1) - eigenValues(0)
			) << UnitHandler::getUnitString<Quantity::Energy>() << "\n";

		return 0;
	}
```
Here we first request the Property::EigenValues from the PropertyExtractor.
This object can be seen as a function of the eigenstate indices and we can obtain the individual energies using *eigenValues(0)* and *eigenValues(1)*.
The object eigenValues also contains information about the number of actual eigenvalues, which is obtained using *eigenValues.getSize()*.
This is used in the newly added for-loop to loop over all eigenvalues and print them.

We make two notes about the expression inside the for-loop.
First, instead of printing the eigenValues immediately, we first pass them through the function UnitHandler::convertNaturalToBase() to convert the values from "natural to base" units.
This is done since all numbers in TBTK are in the natural units specified by the UnitHandler calls at the beginning of the program.
Certainly, it is possible to print the values in the natural scale too, but here we want to print them in the base units meV.
In fact, in this case the conversion is not strictly necessary since the natural energy scale is set to 1 meV, which means that the natural units and the base units are the same.
However, it is good practice to always perform the conversion even if it is known to be trivial since it makes it possible to later change the natural scale without having to change the rest of the code.
Second, after printing the numeric value of the eigenvalues, we call UnitHandler::getUnitString<Quantity::Energy>() to also print a string representation of the energy unit after the energy values.

In the four last lines, the Zeeman split is calculated and similarly printed.

## Calculating the magnetization
To be continued...
