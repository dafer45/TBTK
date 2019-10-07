Manual {#Manual}
======

- @subpage Introduction
- @subpage Overview
- @subpage UnitHandler
- @subpage Indices
- @subpage HoppingAmplitudes
- @subpage Model
- @subpage Solvers
- @subpage PropertyExtractors
- @subpage Properties
- @subpage ImportingAndExportingData
- @subpage Streams
- @subpage Timer
- @subpage FourierTransform
- @subpage Array
- @subpage Plotting

@page Introduction Introduction

# Origin and scope {#OriginAndScope}
TBTK (Tight-Binding ToolKit) originated as a toolkit for solving tight-binding models, but has expanded in scope since then.
Today it is best described as a framework for building applications that solve second-quantized Hamiltonians with discrete indices  

<center>\f$H = \sum_{\mathbf{i}\mathbf{j}}a_{\mathbf{i}\mathbf{j}}c_{\mathbf{i}}^{\dagger}c_{\mathbf{j}} + \sum_{\mathbf{i}\mathbf{j}\mathbf{k}\mathbf{l}}V_{\mathbf{i}\mathbf{j}\mathbf{k}\mathbf{l}}c_{\mathbf{i}}^{\dagger}c_{\mathbf{j}}^{\dagger}c_{\mathbf{k}}c_{\mathbf{l}}\f$.</center>

Even more general interaction terms than above are possible to model, but there is currently limited support for solving interacting models.
This manual is therefore focused on the non-interacting case.

Note that the use of discrete indices does not imply any restrictions on the type of systems that can be modeled.
First, a continuous description can often be turned into a discrete description by the choice of an appropriate basis.
Second, any remaining continuous coordinate anyway has to be discretized before a numerical method is applied.
For example, the Schrödinger equation in a central potential contains three continuous coordinates.
By working in the basis of spherical harmonics, these are reduced to two discrete indices (\f$l \f$ and \f$m\f$) and one remaining continuous coordinate \f$r\f$.

# A quantum mechanics framwork {#AQuantumMechanicsFramework}

TBTK is not a scientific software that can perform a specific type of calculation.
Rather, it is a framework aimed at simplifying the development of code for performing quantum mechanical calculations.
The main focus is on providing general purpose data structures for problems formulated in the language of second quantization.
These data structures are designed with four goals in mind: generality, efficency, readability, and modularity.

<b>Generality</b> means that a data structure captures the general structure of the concept it represents.
It makes it reusable for many different types of problems.

<b>Efficiency</b> means that the overhead associated with using the data structure is small.
Wherever performance is critical, it should be easy to convert it to the data format that is most suited for the given task.
Method developers should have complete freedom to exploit whatever representation of their data that is needed to achieve maximum performance.

<b>Readability</b> means that it allows for highly readable code to be written.
Application developers should be able to focus on physical questions rather than numerical details.

<b>Modularity</b> means that the data structures have limited interdependencies.
This makes it possible to combine them in the many different ways that are required to implement a wide range of applications.
Modularity also allows for individual components to be broken apart whenever detailed control over the numerics is needed.
While application developers are shielded from uneccesary numerical details, the components should be fully transparent to the developer that needs to dig deeper.

Through these principles, TBTK makes it easy to get started implementing numerical calculations.
It provides the efficency of low level languages, while also removing the complexity barrier that otherwise often is associated with C++.
It also aims to make the community involved in developing code for quantum mechanical calculations more integrated.
Enabling and encouraging the sharing of code by providing a unifying framework that makes it possible to reuse other peoples code with minimal effort.

TBTK also provides utilities and build tools that are meant to streamline the build process.
Crucially, special attention is paid to enabling reproducability of scientific results, and [semantic versioning](https://semver.org/spec/v2.0.0.html) is adopted.
Whenever the log is enabled, the version number together with the git hash for the specific commit is written there.
This makes it possible to rerun a calculation at a later time with the exact same version of TBTK.

@link Overview Next: Overview@endlink
@page Overview Overview

# Model, Solvers, Properties, and PropertyExtractors {#ModelSolversPropertiesAndPropertyExtractors}
To carry out a theoretical study, three questions need to be answered:
- What **model** is used to represent the problem?
- What **method** is used to investigate the model?
- What **properties** are going to be calculated?

In a numerical context, this determines the input, the algorithm, and the output, respectively.
In TBTK, the model specification is performed using the @link Model@endlink class, algorithms are implemented in @link Solvers Solver@endlink classes, and a number of @link Properties Property@endlink classes exists to store information about extracted properties.

Typically, most of the computational time is spent executing the algorithm.
The @link Solvers@endlink are therefore where most low level optimization is required.
To shield application developers from uneccesary exposure to the numerical details, the Solvers can be wrapped in so called PropertyExtractors.
These PropertyExtractors allow for Properties to be extracted from the Solvers using an interface that is independent of the Solver.
Together, the @link Model@endlink, the @link PropertyExtractors@endlink, and the @link Properties@endlink allow the application developer to setup and run calculations using code that emphasises the physics of the problem.

The clear separation between the three tasks also makes it possible to change the code accosiated with either of them without having to modify the code of the other two.
This makes it possible to easily try multiple solution methods for the same problem.

# Units {#Units}
No single set of units are the most appropriate in all quantum mechanical problems.
Therefore, TBTK comes with a @link UnitHandler@endlink that gives the application developer the freedom to specify what units to use.

# Auxiliary tasks
TBTK also contains many classes for performing auxiliary tasks such as @link ImportingAndExportingData importing and exporting data@endlink, @link Streams writing logs@endlink, @link Timer profiling code@endlink, and @link Plotting plotting data@endlink.

# Application development {#ApplicationDevelopment}
This manual is aimed at application developers that want to implement calculations that answers specific physical questions.
Together with the @link InstallationInstructions installation instructions@endlink it explains the key concepts that are needed to get started.
For more in depth information, see the API, which can be found under "Classes" in the menu.

# Method development {#MethodDevelopment}
A key purpose of TBTK is to make it possible to develop general purpose code that can be easily shared with the community.
For example, implementing new Solvers, providing standardized models for different types of materials, etc.
Such contributions can either be released as stand alone packages or be pulled into the TBTK framework itself.
If you are interested in this, the manual is a good place to become familiar with the general workflow.
More detailed information is provided in the API, which can be found under "Classes" in the menu.

# Contact
Please do not hesitate to send an email to kristofer.bjornson@second-tech.com if you have any questions or suggestions.

@link UnitHandler Next: UnitHandler@endlink
@page UnitHandler UnitHandler
@link TBTK::UnitHandler See more details about the UnitHandler in the API@endlink

# Units and constants {#UnitsAndConstants}
Most physical quantities have units, but computers work with unitless numbers.
The units can be made implicit by specifying a software wide convention, but no single set of units are the most natural accross all of quantum mechanics.
Therefore, TBTK provides a @link TBTK::UnitHandler UnitHandler@endlink that makes it possible to specify the units that are most natural for the given application.
All numbers that are passed to TBTK functions are assumed to be given in these units.
The UnitHandler also allows for physical constants to be requested in these natural units.

# Base units {#BaseUnits}
The @link TBTK::UnitHandler UnitHandler@endlink borrows its terminology from the SI standard.
Not by forcing the user to work in SI units, but through a clear division of units into base units and derived units.
To understand what this means, consider distance and time.
These are independent quantities that can be measured in meters (m) and seconds (s).
In comparison, velocity is a derived quantity that is defined as distance per time and have the derived unit m/s.
In principle, nothing prevents us from instead consider time to be a quantity that is derived from distance and velocity.
However, the fact remains that only two of these three quantities can be defined independently of each other.
It turns out that in nature only seven quantities can be defined independently of each other.
If we therefore fix seven such quantities and assigning them base units, all other quantities aquire derived units.

The @link TBTK::UnitHandler UnitHandler@endlink defines the base quantities to be temperature, time, length, energy, charge, and count (amount).
The seventh base quantity, angle, is not yet implemented.
This is different from the SI system, which defines base units for mass, current, and luminosity instead of energy, charge, and angle.
The choice to deviate from the SI system by making energy, charge, and angle base quantities is made since these are percieved to be of greater relevance in quantum mechanical calculations.

The @link TBTK::UnitHandler UnitHandler@endlink also deviates from the SI system by only fixing the base quantities and not the base units.
While the SI unit for length is meter (m), the UnitHandler allows the base unit for length to be set to, among other things, meter (m), millimeter (mm), nanometer (nm), and Ångström (Å).
Similarly, Joule (J) and electronvolt (eV) are possible base units for energy, while Coulomb (C) and elementary charge (e) are examples of base units for charge.

### Default base units
| Quantity    | Default base unit  | UnitHandler symbol |
|-------------|--------------------|--------------------|
| Temperature | K (Kelvin)         | Temperature        |
| Time        | s (seconds)        | Time               |
| Length      | m (meter)          | Length             |
| Energy      | eV (electron Volt) | Energy             |
| Charge      | C (Coulomb)        | Charge             |
| Count       | pcs (pieces)       | Count              |

### Available base units
| Quantity    | Available base units                             |
|-------------|--------------------------------------------------|
| Temperature | kK, K, mK, uK, nK                                | 
| Time        | s, ms, us, ns, ps, fs, as                        |
| Length      | m, mm, um, nm, pm, fm, am, Ao                    |
| Energy      | GeV, MeV, keV, eV, meV, ueV, J                   |
| Charge      | kC, C, mC, uC, nC, pC, fC, aC, Te, Ge, Me, ke, e |
| Count       | pcs, mol                                         |

Here Gx, Mx, kx, mx, etc. corresponds to giga, mega, kilo, milli, etc, while Ao means Ångström (Å), and pcs means pieces.
Additonal units can be added on request.

If base units other than the default ones are wanted, this should be specified at the very start of the application.
This avoids ambiguities that otherwise can result from changing base units in the middle of the execution.
To set the base units to C, mol, meV, Å, mK, and ps, type
```cpp
	UnitHandler::setScales({"1 C", "1 mol", "1 meV", "1 Ao", "1 mK", "1 ps"});
```
Note that the units in the brackets has to come in the order charge, count, energy, length, temperature, and time.

# Natural units {#NaturalUnits}
The @link TBTK::UnitHandler UnitHandler@endlink extends the base unit concept to natural units by allowing for a numerical prefactor.
For example, consider the following specification.
```cpp
	UnitHandler::setScales({"1 C", "1 pcs", "13.606 eV", "1 m", "1 K", "1 s"});
```
Here the natural energy unit is set to 13.606 eV.
This means that any energy value that is passed to a TBTK function is interpreted to be given in terms of 13.606 eV.
For example, the numeric value 1.5 is interpreted to mean 1.5*13.606 eV = 20.409 eV.
Note the important distinction between base units and natural units: in base units the value is 20.409 eV, but in natural units it is 1.5.

# Converting between base and natural units {#ConvertingBetweenBaseAndNaturalUnits}
To convert between base units and natural units, the following functions can be used.
```cpp
	double quantityInBaseUnits
		= UnitHandler::convertQuantityNtB(quantityInNaturalUnits);
	double quantityInNaturalUnits
		= UnitHandler::convertQuantityBtN(quantityInBaseUnits);
```
Here 'Quantity' is to be replace by the corresponding UnitHandler symbol specified in the table above, and NtB and BtN should be read 'natural to base' and 'base to natural', respectively.

# Derived units {#DerivedUnits}
To aid conversion between different units, the @link TBTK::UnitHandler UnitHandler@endlink explicitly defines units for a number of derived quantities.
Please, do not hesitate to request additional derived units or quantities if needed.
| Quantity                | Available derived units                      | UnitHandler symbol |
|-------------------------|----------------------------------------------|--------------------|
| Mass                    | kg, g, mg, ug, ng, pg, fg, ag, u             | Mass               |
| Magnetic field strength | MT, kT, T, mT, uT, nT, GG, MG, kG, G, mG, uG | MagneticField      |
| Voltage                 | GV, MV, kV, V, mV, uV, nV                    | Voltage            |

To convert mass in the derived units \f$kg\f$ to and from base units, the follwoing functions can be called.
```cpp
	double massInBaseUnits = UnitHandler::convertMassDtB(
		massInDerivedUnits,
		UnitHandler::MassUnit::kg
	);
	double massInDerivedUnits = UnitHandler::convertMassBtD(
		massInBaseUnits,
		UnitHandler::MassUnit::kg
	);
```
Similarly, it is possible to convert between derived units and natural units.
```cpp
	double massInNaturalUnits = UnitHandler::convertMassDtN(
		massInDerivedUnits,
		UnitHandler::MassUnit::kg
	);
	double massInDerivedUnits = UnitHandler::convertMassNtD(
		massInNaturalUnits,
		UnitHandler::MassUnit::kg
	);
```
Here DtB, BtD, DtN, and NtD should be read as 'derived to base', 'base to derived', 'derived to natural', and 'natural to derived', respectively.
The function calls mimic those for conversion between base and natural units, with the exception that the derived unit has to be passed as a second argument.

# Constants {#Constants}
The following constants can be requested from the @link TBTK::UnitHandler UnitHandler@endlink.
| Name                    | Symbol           | UnitHandler symbol |
|-------------------------|------------------|--------------------|
| Reduced Planck constant | \f$\hbar\f$      | Hbar               |
| Boltzman constant       | \f$k_B\f$        | K_B                |
| Elementary charge       | \f$e\f$          | E                  |
| Speed of light          | \f$c\f$          | C                  |
| Avogadros number        | \f$N_A\f$        | N_A                |
| Electron mass           | \f$m_e\f$        | M_e                |
| Proton mass             | \f$m_p\f$        | M_p                |
| Bohr magneton           | \f$\mu_B\f$      | Mu_B               |
| Nuclear magneton        | \f$\mu_n\f$      | Mu_n               |
| Vacuum permeability     | \f$\mu_0\f$      | Mu_0               |
| Vacuum permittivity     | \f$\epsilon_0\f$ | Epsilon_0          |
Please do not hesitate to request the addition of further constants.

To get a constant value in base or natural units, perform the following calls with 'Symbol' replaced by the corresponding symbol listed in the table above.
```cpp
	double constantValueInBaseUnits    = UnitHandler::getSymbolB();
	double constantValueInNaturalUnits = UnitHandler::getSymbolN();
```

# Unit strings {#UnitStrings}
To aid with printing values to the output, the @link TBTK::UnitHandler UnitHandler@endlink provides methods for requesting the string representation of a given quantity or constants base unit.
```cpp
	string unitSring = UnitHandler::getSymbolUnitString();
```
Here 'Symbol' should be replaced by one of the symbols listed in the tables over base units, derived units, and constants.
Note that the unit string corresponds to the base units, so any output from TBTK should first be converted from natural units to base units before being printed together with the unit string.

@link Indices Next: Indices@endlink
@page Indices Indices
@link TBTK::Index See more details about the Index in the API@endlink

# Physical indices
Physical quantities often carry indices.
For example, the wave function \f$\Psi_{\sigma}(x, y, z)\f$ has three spatial indices \f$x, y, z\f$ and a spin index \f$\sigma\f$.
In TBTK, each such index is referred to as a @link TBTK::Subindex Subindex@endlink and are grouped together into an @link TBTK::Index Index@endlink such as {x, y, z, spin}.
The wave function is therefore said to be indexed by a single Index.
Likewise, the two operators in the Hamiltonian \f$H = \sum_{\mathbf{i}\mathbf{j}}a_{\mathbf{i}\mathbf{j}}c_{\mathbf{i}}^{\dagger}c_{\mathbf{j}}\f$ are each indexed by a single Index, while the coefficient \f$a_{\mathbf{i}\mathbf{j}}\f$ carries two Indices.

An @link TBTK::Index Index@endlink can contain an arbitrary number of Subindices and have an arbitrary structure.
The Indices {x, y, z, spin}, {x, y, sublattice, orbital, spin}, {kx, ky, spin}, and {spin, orbital, x, y} all have valid Index structures.
The only limitation is that the @link TBTK::Subindex Subindices@endlink must be non-negative numbers.

# Creating indices {#CreatingIndices}
An @link TBTK::Index Index@endlink can be created as follows
```cpp
	Index index({x, y, z, spin});
```
However, most of the time Indices enter as function arguments, in which case they commonly appear in code simply as
```cpp
	... {x, y, z, spin} ...
```
Given and Index, it is possible to get its size and compontents using
```cpp
	unsigned int size = index.size();
	int x = index[0];
	int y = index[1];
	int z = index[2];
	int spin = index[3];
```

# Hilbert space indices {#HilbertSpaceIndices}
Algorithms typically work most efficiently when the indices are linear.
For example, if the indices \f$\mathbf{i}\f$ and \f$\mathbf{j}\f$ in the Hamiltonian \f$H = \sum_{\mathbf{i}\mathbf{j}}a_{\mathbf{i}\mathbf{j}}c_{\mathbf{i}}^{\dagger}c_{\mathbf{j}}\f$ are simple row and column indices, it is straight forward to express \f$H\f$ as a matrix and diagonalize it.
A possible linearization of {x, y, z, spin} is h = 2*(sizeX*(sizeY*x + y) + z) + spin.
In TBTK, a linearized index such as h is referred to as a Hilbert space index.

There are two major issues with hard coding a linearization like the one above.
First, it locks the implementation to a particular system, in this case a three-dimensional lattice of size sizeX*sizeY*sizeZ.
Second, it is a mental burden to the developer.
The more intuitive notation {x, y, z, spin} allows for emphasis to be put on physics rather than numerics.

TBTK therefore internally linearizes the indices in a way that is both generally applicable and which generates a minimal Hilbert space size for the given problem.
This allows application developers to specify the @link Model Models@endlink and extract @link Properties@endlink using physical @link TBTK:Index Indices@endlink alone.
It also allows method developers to request the Model in a linear Hilbert space basis.
Application developers can therefore focus on the physics of their particular problem, while method developers can write efficient general purpose @link Solvers@endlink that are free from Model specific assumptions.

# Advanced: Complex index structures {#ComplexIndexStructures}
When seting up a @link TBTK::Model Model@endlink, there are two rules:

- <b>Rule 1:</b> For a given @link TBTK::Index Index@endlink structure, not every Index in a range needs to be included.
- <b>Rule 2:</b> If two Indices differ in value in a Subindex, the Index structure that results from the remaining Subindices to the right of it are allowed to be completely unrelated.

<b>Example 1:</b>
Consider a square lattice with Index structure {x, y} and size 10x10.
By <b>rule 1</b> we are allowed to exclude N lattice points from the @link Model@endlink.
TBTK ensures that a minimal Hilbert space is generated, which in this case will have the size 100 - N.

<b>Example 2:</b>
Consider a molecule on top of a three-dimensional substrate.
Each subsystems may be best described by {x, spin} and {x, y, z, spin}, respectively.
Moreover, the x-@link TBTK::Subindex Subindices@endlink do not need to be related: neither by being aligned along the same direction nor by having the same ranges.
This can be modeled using the Index structures {0, x, spin} and {1, x, y, z, spin}, where a subsystem Subindex has been prepended.
Since the two subsystems differs in value in the first Subindex, <b>rule 2</b> implies that there needs to be no relation between the Subindices to the right of it.

<b>Example 3:</b>
Consider a two-dimensional lattice with a unit cell that contains two different types of atoms.
An appropriate @link TBTK::Index Index@endlink structure may be {x, y, sublattice, orbital, spin}.
By <b>rule 1</b>, there is no problem if the number of orbitals are different for the two different sublattices.<b>*</b>
Moreover, if one of the sublattices only have one orbital, it is possible to drop the orbital @link TBTK::Subindex Subindex@endlink completely for that sublattice.
By <b>rule 2</b>, it is possible to use the Index structures {x, y, 0, orbital, spin} and {x, y, 1, spin}.
More specifically, this is possible since the sublattice Subindex differs in value for the two sublattices and it stands to the left of the orbital Subindex.

<b>*</b> In fact, by <b>rule 1</b> it is also possible for two different sites on the same sublattice to have different number of orbitals, as may be the case in a doped material.

@link HoppingAmplitudes Next: HoppingAmplitudes@endlink
@page HoppingAmplitudes HoppingAmplitudes
@link TBTK::HoppingAmplitude See more details about the HoppingAmplitude in the API@endlink

# Terminology {#Terminology}
Consider the Schrödinger equation
<center>\f$i\hbar\frac{d}{dt}|\Psi(t)\rangle = H|\Psi(t)\rangle\f$,</center>
and the single particle Hamiltonian
<center>\f$H = \sum_{\mathbf{i}\mathbf{j}}a_{\mathbf{i}\mathbf{j}}c_{\mathbf{i}}^{\dagger}c_{\mathbf{j}}\f$.</center>
Using finite differences, the Schrödinger equation can be rewritten as
<center>\f$|\Psi(t+dt)\rangle = \left(1 - i\hbar Hdt\right)|\Psi(t)\rangle\f$.</center>
On this form it is clear that \f$a_{\mathbf{i}\mathbf{j}}\f$ is related to the rate at which single-particle processes move particles from state \f$\mathbf{j}\f$ to state \f$\mathbf{i}\f$.
Due to TBTK's origin in tight-binding calculations, these parameters are called hopping amplitudes.
However, we note that not all \f$a_{\mathbf{i}\mathbf{j}}\f$ corresponds to movement of particles.
For example, \f$a_{\mathbf{i}\mathbf{i}}\f$ is related to the rate at which particles are annihilated and immediately recreated in the same state.
Likewise, if \f$\mathbf{i}\f$ and \f$\mathbf{j}\f$ only differ in a spin index, then the hopping amplitude is related to the rate of a spin flip process.

The indices \f$\mathbf{i}\f$ and \f$\mathbf{j}\f$ are the state in which the particle ends up and starts out, respectively.
Therefore they are referred to as to-indices and from-indices, respectively, in TBTK.

# The HoppingAmplitude class {#TheHoppingAmplitudeClass}
The @link TBTK::HoppingAmplitude HoppingAmplitude@endlink class represents \f$a_{\mathbf{i}\mathbf{j}}\f$ and can be created as
```cpp
	HoppingAmplitude hoppingAmplitude(amplitude, toIndex, fromIndex);
```
Here amplitude is of the type std::complex<double> and toIndex and fromIndex are @link Indices@endlink.
However, typically a HoppingAmplitude is created as an argument to a function with explicit Indices written out, in which case it looks more like
```cpp
	... HoppingAmplitude(amplitude, {x+1, y, spin}, {x, y, spin}) ...
```

Given a @link TBTK::HoppingAmplitude HoppingAmplitude@endlink, it is possible to extracte the amplitude and the to- and from-@link Indices@endlink.
```cpp
	complex<double> amplitude = hoppingAmplitude.getAmplitude();
	const Index &toIndex      = hoppingAmplitude.getToIndex();
	const Index &fromIndex    = hoppingAmplitude.getFromIndex();
```

# Advanced: Callback functions {#AdvancedCallbackFunctions}
Sometimes it is useful to delay the specification of a @link TBTK::HoppingAmplitude HoppingAmplitudes@endlink value.
This is for example the case when a @link Model@endlink is going to be solved multiple times for different parameter values.
Another case is when some of the parameters in the Hamiltonian are to be determined self-consistently.
For this reason, it is possible to pass a callback to the HoppingAmplitude instead of a value.

Consider a problem with the @link Indices Index@endlink structure {x, y, spin} and let \f$J\f$ be a parameter that determines the strength of a Zeeman term.
It is then possible to implement a callback as follows
```cpp
	class ZeemanCallback : public HoppingAmplitude::AmplitudeCallback{
	public:
		complex<double> getHoppingAmplitude(
			const Index &to,
			const Index &from
		) const{
			//Get spin index.
			int spin = from[2];

			//Return the value of the HoppingAmplitude
			return -J*(1 - 2*spin);
		}

		void setJ(complex<double> J){
			this->J = J;
		}
	private:
		double J;
	}
```
Here the function *getHoppingAmplitude()* is responsible for returning the correct value for the given combination of to- and from-Indices.
Note that the spin is read from the from-Index.
This is because it is assumed that this callback will only be used together with HoppingAmplitudes for which the to- and from-Indices are the same.
In general, the implementation of *getHoppingAmplitude()* may involve looking at multiple @link TBTK::Subindex Subindices@endlink of both Indices to figure out what to return.
Finally, the *setJ()* function allows for the J value to be changed.

With the callback defined it is possible to create a callback dependent @link TBTK::HoppingAmplitude HoppingAmplitude@endlink.
```cpp
	ZeemanCallback zeemanCallback;
	HoppingAmplitude up(zeemanCallback, {0, 0, 0}, {0, 0, 0});
	HoppingAmplitude down(zeemanCallback, {0, 0, 1}, {0, 0, 1});

	zeemanCallback.setJ(1);
	Streams::out
		<< up.getAmplitude() << "\t"
		<< down.getAmplitude() << "\n";

	Streams::out << "\n";
	zeemanCallback.setJ(2);
	Streams::out
		<< up.getAmplitude() << "\t"
		<< down.getAmplitude() << "\n";
```
<b>Output:</b>
```bash
	-1
	1

	-2
	2
```

@link Model Next: Model@endlink
@page Model Model
@link TBTK::Model See more details about the Model in the API@endlink

# The Model class {#TheModelClass}
The @link TBTK::Model Model@endlink class is a container for the Hamiltonian, chemical potential, temperature, statistics, etc.
It is best understood through an example, therefore consider a two level system described by the Hamiltonian
<center>\f$H = U_{0}c_{0}^{\dagger}c_{0} + U_{1}c_{1}^{\dagger}c_{1} + V\left(c_{0}^{\dagger}c_{1} + H.c.\right)\f$,</center>
which can be written
<center>\f$H = \sum_{\mathbf{i}\mathbf{j}}a_{\mathbf{i}\mathbf{j}}c_{\mathbf{i}}^{\dagger}c_{\mathbf{j}}\f$,</center>
where \f$a_{00} = U_0\f$, \f$a_{11} = U_1\f$, and \f$a_{01} = a_{10} = V\f$.
Further, assume that we will study the system at temperature T = 300K, chemical potential mu = 0, and for particles with Fermi-Dirac statistics.

Letting U_0, U_1, and V be the numerical symbols for \f$U_0, U_1\f$, and \f$V\f$, we can implement the model as follows.
```cpp
	//Create the Model.
	Model model;

	//Set the temperature, chemical potential, and statistics.
	model.setTemperature(300);
	model.setChemicalPotential(0);
	model.setStatistics(Statistics::FermiDirac);

	//Specify the Hamiltonian.
	model << HoppingAmplitude(U_0, {0}, {0});
	model << HoppingAmplitude(U_1, {1}, {1});
	model << HoppingAmplitude(V,   {0}, {1}) + HC;

	//Construct a linearized basis for the Model.
	model.construct();
```
Here the operator<< is used to pass the @link HoppingAmplitudes@endlink to the @link TBTK::Model Model@endlink.
Also note the use of "+ HC" in the line
```cpp
	model << HoppingAmplitude(V, {0}, {1}) + HC;
```
which is equivalent to
```cpp
	model << HoppingAmplitude(V,       {0}, {1});
	model << HoppingAmplitude(conj(V), {1}, {0});
```
The final call to *model.construct()* creates a @link HilbertSpaceIndices linearized Hilbert space basis@endlink for the Indices passed into the Model.

# A more complex example {#AMoreComplexExample}
To see how a more complex @link TBTK::Model Model@endlink can be set up, consider a magnetic impurity on top of a two-dimensional substrate of size 51x51.
We describe the substrate with the Hamiltonian
<center>\f$H_{S} = U_S\sum_{\mathbf{i}\sigma}c_{\mathbf{i}\sigma}^{\dagger}c_{\mathbf{i}\sigma} - t\sum_{\langle\mathbf{i}\mathbf{j}\rangle\sigma}c_{\mathbf{i}\sigma}^{\dagger}c_{\mathbf{j}\sigma},\f$</center>
where \f$\mathbf{i}\f$ is a two-dimensional index, \f$\sigma\f$ is a spin index, and \f$\langle\mathbf{i}\mathbf{j}\rangle\f$ denotes summation over nearest neighbors.
The impurity is described by
<center>\f$H_{Imp} = (U_{Imp} - J)d_{\uparrow}^{\dagger}d_{\uparrow} + (U_{Imp} + J)d_{\downarrow}^{\dagger}d_{\downarrow},\f$</center>
and is connected to site (25, 25) in the substrate through
<center>\f$H_{Int} = \delta\sum_{\sigma}c_{(25,25)\sigma}^{\dagger}d_{\sigma} + H.c.\f$</center>
The total Hamiltonian is
<center>\f$H = H_{S} + H_{Imp} + H_{Int}\f$.</center>

We first note that an appropriate @link Indices Index@endlink structure is {0, x, y, s} for the substrate and {1, s} for the impurity, where s means spin.
Reading of the Hamiltonian terms above, we can tabulate the hopping amplitues.
| Hopping amplitude                          | Value         | To Index         | From Index       |
|--------------------------------------------|---------------|------------------|------------------|
| \f$a_{(0,x,y,\sigma),(0,x,y,\sigma)}\f$    | \f$U_S\f$     | {0,   x,   y, s} | {0,   x,   y, s} |
| \f$a_{(0,x+1,y,\sigma),(0,x,y,\sigma)}\f$  | \f$-t\f$      | {0, x+1,   y, s} | {0,   x,   y, s} |
| \f$a_{(0,x,y,\sigma),(0,x+1,y,\sigma)}\f$  | \f$-t\f$      | {0,   x,   y, s} | {0, x+1,   y, s} |
| \f$a_{(0,x,y+1,\sigma),(0,x,y,\sigma)}\f$  | \f$-t\f$      | {0,   x, y+1, s} | {0,   x,   y, s} |
| \f$a_{(0,x,y,\sigma),(0,x,y+1,\sigma)}\f$  | \f$-t\f$      | {0,   x,   y, s} | {0,   x, y+1, s} |
| \f$a_{(1,\sigma),(1,\sigma)}\f$            | \f$U_{Imp}\f$ | {1, s}           | {1, s}           |
| \f$a_{(1,\uparrow),(1,\uparrow)}\f$        | \f$-J\f$      | {1, 0}           | {1, 0}           |
| \f$a_{(1,\downarrow),(1,\downarrow)}\f$    | \f$J\f$       | {1, 1}           | {1, 1}           |
| \f$a_{(0,25,25,\sigma),(1,\sigma)}\f$      | \f$\delta\f$  | {0,  25,  25, s} | {1, s}           |
| \f$a_{(1,\sigma),(0,25,25,\sigma)}\f$      | \f$\delta\f$  | {1, s}           | {0,  25,  25, s} |
The left column contains the analytic symbol, while the three other columns contain the corresponding information that is needed to create a @link HoppingAmplitudes HoppingAmplitude@endlink.
Subindices that are not numeric values (or arrows) should be understood to be summed/looped over in the analytic/numeric expressions.

The line pairs (2 and 3), (4 and 5), and (9 and 10) are the Hermitian conjugates of each other.
Further, line 7 and 8 can be combined into a single line if we write the amplitude as -J(1 - 2s).
Taking this into account, the table can be compressed into
| Value            | To Index         | From Index       | Add Hermitian conjugate |
|------------------|------------------|------------------|-------------------------|
| \f$U_S\f$        | {0,   x,   y, s} | {0,   x,   y, s} |                         |
| \f$-t\f$         | {0, x+1,   y, s} | {0,   x,   y, s} | Yes                     |
| \f$-t\f$         | {0,   x, y+1, s} | {0,   x,   y, s} | Yes                     |
| \f$U_{Imp}\f$    | {1, s}           | {1, s}           |                         |
| \f$-J(1 - 2s)\f$ | {1, 0}           | {1, 0}           |                         |
| \f$\delta\f$     | {0,  25,  25, s} | {1, s}           | Yes                     |

It is now straight forward to implement the @link TBTK::Model Model@endlink.
```cpp
	const int SIZE_X = 51;
	const int SIZE_Y = 51;

	double U_S = 1;
	double U_Imp = 1;
	double t = 1;
	double J = 1;
	double delta = 1;

	Model model;

	//Setup the substrate.
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				model << HoppingAmplitude(
					U_S,
					{0, x, y, s},
					{0, x, y, s}
				);

				if(x+1 < SIZE_X){
					model << HoppingAmplitude(
						-t,
						{0, x+1, y, s},
						{0, x,   y, s}
					) + HC;
				}
				if(y+1 < SIZE_Y){
					model << HoppingAmplitude(
						-t,
						{0, x, y+1, s},
						{0, x, y,   s}
					) + HC;
				}
			}
		}
	}

	for(int s = 0; s < 2; s++){
		//Setup the impurity.
		model << HoppingAmplitude(     U_Imp, {1, s}, {1, s});
		model << HoppingAmplitude(-J*(1-2*s), {1, s}, {1, s});

		//Add the coupling between the substrate and the impurity.
		model << HoppingAmplitude(
			delta,
			{0, SIZE_X/2, SIZE_Y/2, s},
			{1, s}
		) + HC;
	}

	//Construct a linearized basis for the Model.
	model.construct();
```

The conversion from the analytic Hamiltonian to the numeric @link TBTK::Model Model@endlink was here done in quite some detail.
However, with a bit of experience it is possible to read of the @link HoppingAmplitudes HoppingAmplitude@endlink parameters immediately from the Hamiltonian without having to write down any tables.

We finish this example with an advice to always utilize the Hermitian conjugate to its maximum as we did above.
Note in particular that we used this to only have \f$x+1\f$ and \f$y+1\f$ in the to-@link Indices Index@endlink (and there are no \f$x-1\f$ or \f$y-1\f$).
In short, only forward hopping terms are explicit.
This improves readability and reduces the chance of introducing errors.

# Advanced: Using IndexFilters to construct a Model {#AdvancedUsingIndexFiltersToConstructAModel}
In the example above, if-statements were added inside the first loop to prevent @link TBTK::HoppingAmplitude HoppingAmplitudes@endlink from being added across the boundary.
The code can be made more readable by using an @link TBTK::AbstractIndexFilter IndexFilter@endlink, which allows for the handling of such exceptions to be separated from the main @link TBTK::Model Model@endlink specification.

We here demonstrate how an @link TBTK::AbstractIndexFilter IndexFilter@endlink can be used to create a square shaped geometry with a hole inside.
To do so, we begin by defining the IndexFilter.
```cpp
class Filter : public AbstractIndexFilter{
public:
	Filter(int sizeX, int sizeY, double radius){
		this->sizeX = sizeX;
		this->sizeY = sizeY;
		this->radius = radius;
	}

	Filter* clone() const{
		return new Filter(sizeX, sizeY, radius);
	}

	bool isIncluded(const Index &index){
		double r = sqrt(
			pow(index[0] - sizeX/2, 2)
			+ pow(index[1] - sizeY/2, 2)
		);
		if(r < radius)
			return false;
		else if(index[0] < 0 || index[0] >= sizeX)
			return false;
		else if(index[1] < 0 || index[1] >= sizeY)
			return false;
		else
			return true;
	}
private:
	int sizeX, sizeY;
	double radius;
};
```
The Filter needs to know the size of the the system and the holes radius, therefore the constructor takes these as arguments and stores them in its private variables.
The *clone()*-function is required and should return a pointer to a copy of the Filter.
Finally, the *isIncluded()*-function is responsible for returning true or false depending on whether the given @link Indices Index@endlink should be included in the @link TBTK::Model Model@endlink or not.
In this example, it first checks whether the Index is inside the excluded radius and returns false if this is the case.
Otherwise, it returns true or false depending on whether or not the Index is inside the range [0, sizeX)x[0, sizeY).

With the Filter defined, a @link TBTK::Model Model@endlink with the size 51x51 and a hole radius of 10 can be set up as follows.
```cpp
	Model model;
	model.setFilter(Filter(51, 51, 10));
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				model << HoppingAmplitude(
					U_s,
					{x, y,   s},
					{x, y,   s}
				);
				model << HoppingAmplitude(
					-t,
					{x+1, y, s},
					{x,   y, s}
				) + HC;
				model << HoppingAmplitude(
					-t,
					{x, y+1, s},
					{x, y,   s}
				) + HC;
			}
		}
	}
	model.construct();
```
Note the call to *model.setFilter()* and the fact that no if-statements appear inside the loop.

We end this section with a note about substems with @link ComplexIndexStructures complex Index strucures@endlink.
Consider a @link TBTK::Model Model@endlink with two subsystems Indexed by {0, x, spin} and {1, x, y, z, spin}, respectively.
The @link TBTK::AbstractIndexFilter IndexFilter@endlink needs to be able to handle both of these Index structures.
For example, requesting *index[4]* in the *isIncluded()*-function before knowing that the Index belongs to subsystem 1 is an error.
Therefore, an appropriate implementation of the *isIncluded()*-function would in this case look something like this.
```cpp
	bool isIncluded(const Index &index){
		int subsystem = index[0];
		switch(subsystem){
		case 0:
			int x = index[1];
			int spin = index[2];
			//Do the relevant checks for subsystem 0 here.
			//...

			break;
		case 1:
			int x = index[1];
			int y = index[2];
			int z = index[3];
			int spin = index[4];
			//Do the relevant checks for subsystem 1 here.
			//...

			break;
		}
	}
```

# Block structure {#BlockStructure}
The @link TBTK::Model Model@endlink can recognize a block diagonal Hamiltonian under the condition that the block diagonal @link TBTK::Subindex Subindices@endlink appear to the left in the @link Indices Index@endlink.
Consider a translationally invariant system that is block-diagonal in kx, ky, and kz, but not in orbital or spin.
If the Index structure {kx, ky, kz, orbital, spin} is used, the Model will be able to recognize the block structure.
This can be utilized by some @link Solvers@endlink such as the @link SolverBlockDiagonalizer BlockDiagonalizer@endlink to speed up calculations.
If instead {kx, ky, orbital, kz, spin} is used, only the kx and ky Subindices will be recognized as block indices.

@link Solvers Next: Solvers@endlink
@page Solvers Solvers
@link TBTK::Solver::Solver See more details about the Solvers in the API@endlink

# Solvers and PropertyExtractors
Algorithms for solving @link Model Models@endlink are implemented in @link TBTK::Solver Solvers@endlink.
Since this typically is where most of the computational time is spent, low level optimization often is required.
The purpose of the Solver is to contain these numerical details and present an intuitive interface to the application developer.
The interface allows for the algorithm to be configured on demand, but aims to minimize the amount of method specific knowledge that is necessary to use the Solver.

Internally, the @link TBTK::Solver Solver@endlink converts the @link Model@endlink to the format that is best suited for the algorithm.
Likewise, the output of the algorithm is stored on a method specific format.
This data can be requested from the Solver, but the Solver is not responsible for providing the output on a method independent format.
Instead, each Solver can be wraped in a corresponding @link PropertyExtractors PropertyExtractor@endlink, through which @link Properties@endlink can be extracted using a notation that is Solver independent.
In this chapter, we therefore only discuss how to set up, configure, and run a Solver, leaving the extraction of Properties to the @link PropertyExtractors next chapter@endlink.

# Overview of native Solvers {#OverviewOfNativeSolvers}
TBTK currently contains four production ready @link TBTK::Solver::Solver Solvers@endlink: the @link TBTK::Solver::Diagonalizer Diagonalizer@endlink, @link TBTK::Solver::BlockDiagonalizer BlockDiagonalizer@endlink, @link TBTK::Solver::ArnoldiIterator ArnoldiIterator@endlink, and @link TBTK::Solver::ChebyshevExpander ChebyshevExpander@endlink.
All of these are single-particle Solvers, which is the reason that this manual at the moment only describes how to set up and solve single-particle problems.

The two first @link TBTK::Solver Solvers@endlink are based on diagonalization and can calculate all eigenvalues and eigenvectors.
Their strength is that they provide solutions with complete information about the system, allowing for arbitrary @link Properties@endlink to be calculated.
However, diagonalization scales poorly with system size and is therefore not feasible for very large systems.
If the Model is @link BlockStructure block diagonal@endlink, the @link TBTK::Solver::BlockDiagonalizer BlockDiagonalizer@endlink can handle much larger systems than the @link TBTK::Solver::Diagonalizer Diagonalizer@endlink.

The @link TBTK::Solver::ArnoldiIterator ArnoldiIterator@endlink also calculates eigenvalues and eigenvectors, but can be used to calculate only a few of them.
Arnoldi iteration is a so called Krylov method that iteratively builds a larger and larger subspace of the total Hilbert space.
It then performs diagonalization on this restricted subspace.
If only a few eigenvalues or eigenvectors are needed, they can be calculated quickly even if the Model is very large.

The @link TBTK::Solver::ChebyshevExpander ChebyshevExpander@endlink can calculate the Green's function \f$G_{\mathbf{i}\mathbf{j}}(E)\f$ for any given pair of @link Indices@endlink \f$\mathbf{i}\f$ and \f$\mathbf{j}\f$.
It can be used to calculate @link Properties@endlink for very large systems when they are needed for a limited number of Indices.
However, the method is iterative, and infinite precission along the energy axis requires an infinite number of steps.

# General remarks about Solvers {#GeneralRemarksAboutSolvers}
All Solvers inherit from the base class @link TBTK::Solver::Solver@endlink.
This is a very simple class that only implements two functions for setting and getting the @link Model@endlink to solve.
```cpp
	Solver::Solver solver;
	solver.setModel(model);
	const Model &referenceToModel = solver.getModel();
```
It is important that the Model that is passed to the Solver remains in memory for the full lifetime of the Solver.
Therefore, do not call *solver.setModel()* with a temporary Model object.

@link TBTK::Solver::Solver Solvers@endlink are also @link TBTK::Communicator Communicators@endlink, which means that they can output diagnostic information that is possible to mute.
To turn on the output, call
```cpp
	solver.setVerbose(true);
```

# Solver::Diagonalizer {#SolverDiagonalizer}
The @link TBTK::Solver::Diagonalizer Diagonalizer@endlink sets up a dense matrix representing the Hamiltonian and diagonalizes it to obtain the eigenvalues and eigenvectors.
Because of its simplicity, it is a good choice for @link Model Models@endlink with a Hilbert space size up to a few thousands.
Setting it up and running the diagonalization is straight forward.
```cpp
	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();
```
The Solver is now ready to be wrapped in its @link PropertyExtractors PropertyExtractor@endlink to allow for Properties to be extracted.

## Space and time costs
The required amount of space is particularly easy to estimate for the @link TBTK::Diagonalizer Diagonalizer@endlink.
Internally the Hamiltonian is stored as an upper triangular matrix of type std::complex<double> (16 bytes per entry).
The space required to store a Hamiltonian with basis size N is therefore roughly \f$16(N^2/2) = 8N^2\f$ bytes.
The eigenvectors and eigenvalues requres another \f$16N^2\f$ and \f$8N\f$ bytes, respectively.
This adds up to about \f$24N^2\f$ bytes, which runs into the GB range around a basis size of 7000.

The time it takes to diagonalize a matrix cannot be estimated with the same precision since it depends on clock speed etc.
However, as of 2018 it runs into the hour range for basis sizes of a few thousands.
Knowing the execution time for a specific size and computer, the time required for a different size can be accurately predicted using that the time scales as \f$O(N^3)\f$.

## Advanced: Self-consistency callback
In a self-consistent procedure, some input parameter is unknown but can be calculated as an output.
It is therefore possible to make an initial guess, calculate the parameter, and feed the result back in as input.
If this procedure is repeated until (and if) the input and output converge (within a given tollerance), the solution is said to be self-consistent.

The @link TBTK::Solver::Diagonalizer Diagonalizer@endlink is able to execute such a self-consistency loop.
However, what parameter that is to be solved for is problem specific, and so is the meassure of tollerance.
Therefore, after the Hamiltonian has been diagonalized, the Diagonalizer can give control back to the application developer through a self-consistency callback.
The callback is required to calculate the relevant parameter, check for convergence, and return true or false depending on whether or not convergence has been reached.
If the callback returns true, the Diagonalizer finishes, otherwise the Hamiltonian is rediagonalized and the procedure is repeated.

A self-consistency callback is implemented as follows.
```cpp
class SelfConsistencyCallback :
	public Solver::Diagonalizer::SelfConsistencyCallback
{
public:
	bool selfConsistencyCallback(Solver::Diagonalizer &solver){
		//Calculate and update the parameter(s).
		//...

		//Determine whether the solution has converged or not.
		//...

		//Return true if the solution has converged, otherwise return
		//false.
		if(hasConverged)
			return true;
		else
			return false;
	}
};
```

We note that for the self-consistency procedure to work, those @link HoppingAmplitudes@endlink that depend on the parameter must themselves be dependent on a @link AdvancedCallbackFunctions HoppingAmplitude callback@endlink.
It is the application developers responsibility to make sure that the parameter that is calculated in the self-consistency callback is passed on to the HoppingAmplitude callback.

With the callback defined, the self-consistent calculation can be set up.
```cpp
	Solver::Diagonalizer solver;
	solver.setModel(model);
	SelfConsistencyCallback selfConsistencyCallback;
	solver.setSelfConsistencyCallback(selfConsistencyCallback);
	solver.setMaxIterations(100);
	solver.run();
```
The call to *solver.setMaxIterations(100)* caps the number of iterations at 100.
If the calculation has not converged by then it finishes anyway.
For a complete example of a self-consistent calculation, the reader is referred to the SelfConsistentSuperconductivity template in the Templates folder.

# Solver::BlockDiagonalizer {#SolverBlockDiagoanlizer}
The @link TBTK::Solver::BlockDiagonalizer BlockDiagonalizer@endlink mimics the @link TBTK::Solver::Diagonalizer Diagonalizer@endlink, but is able to take advantage of a Hamiltonians @link BlockStructure block structure@endlink.
```cpp
	Solver::BlockDiagonalizer solver;
	solver.setModel(model);
	solver.run();	
```

## Advanced: Self-consistency callback
The implementation of a compatible self-consistency callback for the @link TBTK::Solver::BlockDiagonalizer BlockDiagonalizer@endlink also closely parallels that of the @link TBTK::Solver::Diagonalizer Diagonalizer@endlink.
```cpp
class SelfConsistencyCallback :
	public Solver::BlockDiagonalizer::SelfConsistencyCallback
{
public:
	bool selfConsistencyCallback(Solver::BlockDiagonalizer &solver){
		//Calculate and update the parameter(s).
		//...

		//Determine whether the solution has converged or not.
		//...

		//Return true if the solution has converged, otherwise return
		//false.
		if(hasConverged)
			return true;
		else
			return false;
	}
};
```

# Solver::ArnoldiIterator {#SolverArnoldiIterator}
The @link TBTK::Solver::ArnoldiIterator ArnoldiIterator@endlink allows for a few eigenvalues and eigenvectors to be calculated quickly even if the system is very large.
Starting with the Hamiltonian \f$H\f$ and a random vector \f$v_0\f$, it iteratively calculates \f$v_n\f$ for higher \f$n\f$.
It does so by first calculating \f$\tilde{v}_n = Hv_{n-1}\f$ and then orthonormalizing it against \f$v_{n-1}, ..., v_{0}\f$ to obtain \f$v_n\f$.
The product \f$Hv_{n-1}\f$ amplifies the components of \f$v_{n-1}\f$ that are eigenvectors with extremal eigenvalues.
Therefore, the subspace spanned by \f$\{v_{n}\}\f$ quickly approximates the part of the Hilbert space that contains these extremal eigenvalues.
Finally, performing diagonalization on this much smaller subspace, the extremal eigenvalues and eigenvectors are calculated quickly.
The eigenvalues and eigenvectors obtained in this way are called Ritz-values and Ritz-vectors.

The @link TBTK::Solver::ArnoldiIterator ArnoldiIterator@endlink can be set up as follows.
```cpp
	Solver::ArnoldiIterator solver;
	solver.setModel(model);
	solver.setNumLanczosVectors(200);
	solver.setMaxIterations(500);
	solver.setNumEigenVectors(100);
	solver.setCalculateEigenVectors(true);
	solver.run();
```
Here, *solver.setNumLanczosVectors()* sets the size of the subspace that is to be generated, while *solver.setNumEigenVectors()* determines the number of eigenvalues to calculate.
The number of eigenvalues should at most be as large as the number of Lanczos vectors.
However, since the iteration starts with a random vector, chosing a smaller number of eigenvalues results in the less accurate subspace to be ignored.
Finally, the @link TBTK::Solver::ArnoldiIterator ArnoldiIterator@endlink uses a modified version of the procedure described above called implicitly restarted Arnoldi iteration (see the [ARPACK](https://www.caam.rice.edu/software/ARPACK/) documentation).
This leads to more iterations being executed than the number of Lanczos vectors finally generated.
The line *solver.setMaxIterations()* puts a cap on the number of iterations.

## Shift and invert (extracting non-extremal eigenvalues)
The @link TBTK::Solver::ArnoldiIterator ArnoldiIterator@endlink can also be used to calculate eigenvalues and eigenvectors around a given value \f$\lambda\f$.
This is done by replacing \f$H\f$ by \f$(H - \lambda)^{-1}\f$, which is referred to as shift-and-invert.
To perform shift-and-invert, we need to set the "central value" \f$\lambda\f$ and instruct the ArnoldiIterator to work in this mode.
```cpp
	solver.setCentralValue(1);
	solver.setMode(Solver::ArnoldiIterator::Mode::ShiftAndInvert);
```

# Solver::ChebyshevExpander {#SolverChebyshevSolver}
The @link TBTK::Solver::ChebyshevExpander ChebyshevExpander@endlink calculates the Green's function on the form
<center>\f$G_{\mathbf{i}\mathbf{j}}(E) = \frac{1}{\sqrt{s^2 - E^2}}\sum_{m=0}^{\infty}\frac{b_{\mathbf{i}\mathbf{j}}^{(m)}}{1 + \delta_{0m}}F(m\textrm{acos}(E/s))\f$.</center>
Here \f$F(x)\f$ is one of the functions \f$\cos(x)\f$, \f$\sin(x)\f$, \f$e^{ix}\f$, and \f$e^{-ix}\f$.
Further, the "scale factor" \f$s\f$ must be chosen large enough that all of the Hamiltonians eigenvalues fall inside of the range \f$|E/s| < 1\f$. 
This is called a Chebyshev expansion of the Green's function and for more details the reader is referred to: Phys. Rev. Lett. <b>78</b>, 275 (2006), Phys. Rev. Lett. <b>105</b>, 1 (2010), and <a href="http://urn.kb.se/resolve?urn=urn%3Anbn%3Ase%3Auu%3Adiva-305212">urn:nbn:se:uu:diva-305212</a>.

The @link TBTK::Solver::ChebyshevExpander ChebyshevExpander@endlink calculates the Green's function in two steps.
In the first step, the expansion coefficients
<center>\f{eqnarray*}{
	b_{\mathbf{i}\mathbf{j}}^{(m)} &=& \langle j_{\mathbf{i}}^{(0)}|j_{\mathbf{j}}^{(m)}\rangle,
\f}</center>
are calculated using the recursive expressions
<center>\f{eqnarray*}{
	|j_{\mathbf{j}}^{(1)}\rangle &=& H|j_{\mathbf{j}}^{(0)}\rangle,\\
	|j_{\mathbf{j}}^{(m)}\rangle &=& 2H|j_{\mathbf{j}}^{(m-1)}\rangle - |j_{\mathbf{j}}^{(m-2)}\rangle.
\f}</center>
Here \f$|j_{\mathbf{i}}^{(0)}\rangle\f$ and \f$|j_{\mathbf{j}}^{(0)}\rangle\f$ are vectors with zeros everywhere, except for a one in the position associated with \f$\mathbf{i}\f$ and \f$\mathbf{j}\f$, respectively.
Using sparse matrix-vector multiplication, these expansion coefficients can be calculated efficiently.
However, an exact solution requires an infinite number of coefficients and a cutoff must therefore be introduced.

In the second step, the @link TBTK::Solver::ChebyshevExpander ChebyshevExpander@endlink generates the Green's function using the calculated coefficients and the expression for the Green's function above.

## Setting up and configuring the solver
A @link TBTK::Solver::ChebyshevExpander ChebyshevExpander@endlink with \f$s = 10\f$ and the number of expansion coefficients capped at 1000 can be set up as follows.
```cpp
	Solver::ChebyshevExpander solver;
	solver.setModel(model);
	solver.setScaleFactor(10);
	solver.setNumCoefficients(1000);
```

The expansion coefficeints can be efficiently calculated on a GPU.
```cpp
	solver.setCalculateCoefficientsOnGPU(true);
```

Also the Green's functions can be generated on a GPU (but this is less important and runs into memory limitations relatively quickly).
```cpp
	solver.setGenerateGreensFunctionsOnGPU(true);
```

When the Green's function is generated for more than one pair of Indices, the execution time can be significantly decreased by using a precalculated lookup table.
```cpp
	solver.setUseLookupTable(true);
```
The lookup table is required to generate the Green's function on a GPU and the only downside of generating it is that it can consume large amounts of memory.

@link PropertyExtractors Next: PropertyExtractors@endlink
@page PropertyExtractors PropertyExtractors
@link TBTK::PropertyExtractor::PropertyExtractor See more details about the PropertyExtractors in the API@endlink

# Physical properties, not numerics {#PhysicalPropertiesNotNumerics}
In order to allow application developers to focus on relevant physical questions rather than algorithm specific details, and to prevent algorithm specific requirements from spreading to other parts of the code, TBTK encourages the use @link TBTK::PropertyExtractor::PropertyExtractor PropertyExtractors@endlink for extracting physical quantities from the @link Solvers@endlink.
PropertyExtractors are interfaces to Solvers that largely present themselves uniformly to other parts of the code.
What this means is that code that relies on calls to a PropertyExtractor is relatively insensitive to what specific Solver that is being used.
The application developer is therefore relatively free to change Solver at any stage in the development process.
This is e.g. very useful when it is realized that a particular Solver is not the best one for the task.
It is also very useful when setting up complex problems where it can be useful to benchmark results from different Solvers against each other.
The later is especially true during the development of new Solvers.

The different @link TBTK::PropertyExtractor::PropertyExtractor PropertyExtractors@endlink can, however, not have completely identical interfaces, since some properties are simply not possible to calculate with some @link Solvers@endlink.
Some Solvers may also make it possible to calculate very specific things that are not possible to do with any other Solver.
The PropertyExtractors are therefore largely uniform interfaces, but not identical.
However, for most standard properties there at least exists function calls that allow the properties to compile even if they cannot actually perform the calculation.
The program will instead print error messages that make it clear that the particular Solver is not able to calculate the property and ask the developer to switch Solver.
In fact, this is achieved through inheritance from a common abstract base class called PropertyExtractor::PropertyExtractor and allows for completely Solver independent code to be written that works with the abstract base class rather than the individual Solver specific PropertyExtractors.
The experienced C++ programmer can use this to write truly portable code, while the developer unfamiliar with inheritance and abstract classes do not need to worry about these details.

Each of the @link Solvers@endlink described in the Solver chapter have their own @link TBTK::PropertyExtractor::PropertyExtractor PropertyExtractor@endlink called @link TBTK::PropertyExtractor::Diagonalizer PropertyExtractor::Diagonalizer@endlink, @link TBTK::PropertyExtractor::BlockDiagonalizer PropertyExtractor::BlockDiagonalizer@endlink, @link TBTK::PropertyExtractor::ArnoldiIterator PropertyExtractor::ArnoldiIterator@endlink, and @link TBTK::PropertyExtractor::ChebyshevExpander PropertyExtractor::ChebyshevExpander@endlink.
Using the Solver::Diagonalizer as an example, the corresponding PropertyExtractor is created using
```cpp
	PropertyExtractor::Diagonalizer propertyExtractor(solver);
```

# Extracting Properties {#ExtractingProperties}
In addition to the @link TBTK::PropertyExtractor::PropertyExtractor PropertyExtractors@endlink, TBTK has a set of @link Properties Property@endlink classes that are returned by the PropertyExtractors and which are more extensively described in the chapter Properties.
These Property classes supports a few different storage modes internally which allows for different types of extraction.
For example does the system in question often have some concrete structure such as a square lattice.
In this case it is useful for properties to preserve knowledge about this structure as it can allow for example two-dimensional plots of the data to be done simply.
Other times no such structure exists, or properties are just wanted for a few different points for which there is no unifying structure.
These different cases require somewhat different approaches for storing the data in memory, as well as for how to instruct the PropertyExtractors how to extract the data.
We here describe how to extract the different properties and the reader can jump to any Property of interest to see how to handle the particular situation.
The reader is, however, advised to first read the first section about the density since this establishes most of the basic notation.
The reader is also referred to the Properties chapter where more details about the Properties are given.

Before continuing, we note that some @link Properties@endlink have an energy dependence.
This means that the quantities needs to be evaluated at a certain number of energy points.
The @link TBTK::PropertyExtractor::PropertyExtractor PropertyExtractors@endlink extracts such properties within an energy window using some energy resolution and this can be set using
```cpp
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);
```
Here the two first numbers are real values satisfying LOWER_BOUND < UPPER_BOUND, and RESOLUTION is an integer specifying the number of energy points that the window is divided into.

## Density
To demonstrate two different modes for extracting properties we consider a @link Model@endlink with the @link Indices Index@endlink-structure {x, y, z, s} with dimensions SIZE_X, SIZE_Y, SIZE_Z, and two spin species.
Next, assume that we are interested in extracting the electron density in the z = 10 plane.
We can do this as follows
```cpp
	Property::Density density = propertyExtractor.calculateDensity(
		{ IDX_X,  IDX_Y,     10, IDX_SUM_ALL},
		{SIZE_X, SIZE_Y, SIZE_Z,           2}
	);
```
Here the first curly brace specifies how the different subindices are to be treated by the @link TBTK::PropertyExtractor::PropertyExtractor PropertyExtractor@endlink.
In this case we specify that the x and y indices are to be considered as a first and second running index.
Note that the labels IDX_X and IDX_Y has nothing to do with the fact that the index structure has x and y variables at these positions.
The two labels could be interchanged, in which case the y-subindex is going to be considered the first index in the @link Properties Property@endlink.
A third specifier IDX_Z is also available and it is important that IDX_Z only is used if IDX_Y is used, and IDX_Y only is used if IDX_X is used.
The third subindex in the first bracket specifies that the PropertyExtractor should only extract the density for z=10.
Finally, the identifier in the fourth position instructs the PropertyExtractor to sum the contribution from all spins.

The second bracket specifies the range over which the subindices run, assuming that they start at {0, 0, 0, 0}.
In this case the third subindex will not actually be used and can in principle be set to any value, for which 1 is another reasonable choice as a reminder that only one value is going to be used.
While there currently is no way of changing the lower bound for the range, it is possible to limit the upper bound by for example passing {SIZE_X/2, SIZE_Y, SIZE_Z, 2} as second argument.
In this case the density will only be evaluated for the lower half of the x-range.

Now assume that we instead are interested in extracting the density for the z = 10 plane, the points along the line (y,z)=(5,15), and the spin down density on site (x,y,z)=(0,0,0).
This can be achieved by passing a list of patterns to the @link TBTK::PropertyExtractor::PropertyExtractor PropertyExtractor@endlink as follows
```cpp
	Property::Density density = propertyExtractor.calculateDensity({
		{_a_, _a_, 10, IDX_SUM_ALL},
		{_a_,   5, 15, IDX_SUM_ALL},
		{  0,   0,  0,           1}
	});
```
First note the two curly brackets on the first and last line which means that the other brackets are passed to the function as a list of brackets rather than as individual arguments.
This allows for an arbitrary number of patterns to be passed to the PropertyExtractor.
The distinction becomes particularly important to keep in mind when only two patterns are supplied, since forgetting the outer brackets will result in the first mode described above to be executed instead.
The symbol \_a\_ indicates wildcards, meaning that any Index that matches the patter will be included independently of the values in these positions.
We note here that while the three underscores are useful for improving readability in application code, it is also possible to use the more descriptive identifier IDX_ALL.

## DOS
The density of states (@link TBTK::Property::DOS DOS@endlink) is representative for a third internal storage mode since being a system wide property it has no Index-structure.
```cpp
	Property::DOS dos = propertyExtractor.calculateDOS();
```

## LDOS
Assuming the index structure {x, y, z, s}, with dimensions SIZE_X, SIZE_Y, SIZE_Z, and two spin species, the @link TBTK::Property::LDOS LDOS@endlink can be extracted for the z = 10 plane as
```cpp
	Property::LDOS ldos = propertyExtractor.calculateLDOS(
		{ IDX_X,  IDX_Y,     10, IDX_SUM_ALL},
		{SIZE_X, SIZE_Y, SIZE_Z,           2}
	);
```
or for the plane z=10, along the line (y,z)=(5,15), and for the down spin on site (x,y,z)=(0,0,0) using
```cpp
	Property::LDOS ldos = propertyExtractor.calculateLDOS({
		{_a_, _a_, 10, IDX_SUM_ALL},
		{_a_,   5, 15, IDX_SUM_ALL},
		{  0,   0,  0,           1}
	});
```

## Magnetization
Assuming the index structure {x, y, z, s}, with dimensions SIZE_X, SIZE_Y, SIZE_Z, and two spin species, the @link TBTK::Property::Magnetization Magnetization@endlink can be extracted for the z = 10 plane as
```cpp
	Property::Magnetization magnetization
		= propertyExtractor.calculateMagnetization(
			{ IDX_X,  IDX_Y,     10, IDX_SPIN},
			{SIZE_X, SIZE_Y, SIZE_Z,        2}
		);
```
or for the plane z=10, along the line (y,z)=(5,15), and for the site (x,y,z)=(0,0,0) using
```cpp
	Property::Magnetization magnetization
		= propertyExtractor.calculateMagnetization({
			{_a_, _a_, 10, IDX_SPIN},
			{_a_,   5, 15, IDX_SPIN},
			{  0,   0,  0, IDX_SPIN}
		});
```
Note that in order to calculate the Magnetization, it is necessary to specify one and only one spin-subindex using IDX_SPIN.

## SpinPolairzedLDOS
Assuming the index structure {x, y, z, s}, with dimensions SIZE_X, SIZE_Y, SIZE_Z, and two spin species, the @link TBTK::Property::SpinPolarizedLDOS SpinPolarizedLDOS@endlink can be extracted for the z = 10 plane as
```cpp
	Property::SpinPolarizedLDOS spinPolarizedLDOS
		= propertyExtractor.calculateSpinPolarizedLDOS(
			{ IDX_X,  IDX_Y,     10, IDX_SPIN},
			{SIZE_X, SIZE_Y, SIZE_Z,        2}
		);
```
or for the plane z=10, along the line (y,z)=(5,15), and for the site (x,y,z)=(0,0,0) using
```cpp
	Property::SpinPolarizedLDOS spinPolarizedLDOS
		= propertyExtractor.calculateSpinPolarizedLDOS({
			{_a_, _a_, 10, IDX_SPIN},
			{_a_,   5, 15, IDX_SPIN},
			{  0,   0,  0, IDX_SPIN}
		});
```
Note that in order to calculate the SpinPolarizedLDOS, it is necessary to specify one and only one spin-subindex using IDX_SPIN.

## Further Properties
Further @link Properties@endlink such as @link TBTK::Property::EigenValues EigenValues@endlink, @link TBTK::Property::GreensFunction GreensFunction@endlink, @link TBTK::Property::SelfEnergy SelfEnergy@endlink, and @link TBTK::Property::WaveFunctions WaveFunctions@endlink are also available but are not yet documented in this manual.
If you are interested in these quantities, do not hesitate to contact kristofer.bjornson@second-tech.com to get further details or to request a speedy update about one or several of these Properties.

@link Properties Next: Properties@endlink
@page Properties Properties
@link TBTK::Property::AbstractProperty See more details about the Properties in the API@endlink

# Properties and meta data {#PropertiesAndMetaData}
When calculating physical properties it is common to store the result in an arrays.
The density at (x,y) for a two-dimensional grid with dimensions (SIZE_X, SIZE_Y) can for example be stored as the array element *density[SIZE_Y*x +y]*.
However, there are two problems with using such a simple storage scheme.
First, there is an implicit assumption in the way the elements are laid out in memory that is nowhere documented in actual code.
Every time the developer needs to write new code that access an element in the array, it is up to the developer to remember that the offset to the element should be calculated as *SIZE_Y*x + y*.
The rule is certainly easy for grid like systems like in this example, but generalizes poorly to complex structures, and moreover limits the possibility to write general purpose functions that takes the array as input.
Second, the variables SIZE_X and SIZE_Y needs to be stored separately from the array and either be global variables or be passed independently to any function that uses the array.

The variables SIZE_X and SIZE_Y, as well as the information that the offset should be calculated as SIZE_Y*x + y, is meta data that together with the data itself forms a self contained concept.
In TBTK properties are therefore stored in @link TBTK::Property::AbstractProperty Property@endlink classes which acts as containers of both the data itself, as well as the relevant meta data.
Moreover, the Properties can internally store the data in multiple different storage modes, each suitable for different types of data.
In this chapter we describe these different storage modes, as well as the various specific properties natively supported by TBTK.
We also note that while this chapter describes the properties themselves, the reader is referred to the @link PropertyExtractors PropertyExtractor@endlink chapter for information about how to actually create the various Properties.

# Storage modes {#StorageModes}
There currently exists three different storage modes known as None, Ranges, and Custom.
The names correspond to the type of Index structures that they are meant for.

## None
The storage mode None is the simplest one and is meant for @link TBTK::Property::AbstractProperty Properties@endlink that has no @link Indices Index@endlink structure at all, which is typical of global properties such as the density of states (@link TBTK::Property::DOS DOS@endlink) or @link TBTK::Property::EigenValues EigenValues@endlink.

## Ranges
The Ranges storage mode is the storage mode described in the first section of this chapter and is meant for Properties that are extracted on a regular grid.
By explicitly preserving the grid structure in the @link TBTK::Property::AbstractProperty Property@endlink, other routines can make stronger assumptions about the data than it otherwise would be able to do, which can be useful in certain cases.
This is particularly true when plotting data, since for example a density extracted on some specific two-dimensional plane in a three-dimensional grid can be plotted as a surface plot.
In contrast, it is not clear how to plot a density extracted from a few randomly chosen points in the three-dimensional grid.
If a common storage format that support the later possibility is chosen also in the former case, additional information will have to be provided to for example a plotter routine to tell it that it actually is more structured than it appears from the storage format alone.
In particular, TBTK comes prepared with python scripts ready to plot many Properties, and many of these only work when the Ranges format is used.

Sometimes it is useful to access the raw data rather than the @link TBTK::Property::AbstractProperty Property@endlink object as a whole.
This can be done as follows
```cpp
	const vector<DataType> &data = property.getData();
```
Here DataType should be replaced by the specific data type of the property.
There also exists a corresponding call that gives write access to the data, but it is recommended to only use this when really needed.
```cpp
	//Warning! Only use this if it is really needed.
	vector<DataType> &data = property.getData();
```

When @link TBTK::Property::AbstractProperty Properties@endlink are extracted on the Ranges format, identifiers IDX_X, IDX_Y, and IDX_Z and corresponding ranges SIZE_X, SIZE_Y, and SIZE_Z are used (see the @link PropertyExtractors@endlink chapter).
These are used to indicate which subindex that should be mapped to the first, second, and third index in the array, and their ranges.
The ranges are stored in the Property and can be accessed using
```cpp
	vector<int> ranges = property.getRanges();
```
Individual elements are then accessed from the array using
```cpp
	data[NUM_INTERNAL_ELEMENTS*(ranges[2]*(ranges[1]*x + y) + z) + n];
```
where x, y, and z corresponds to the first second and third index, respectively.
Further, NUM_INTERNAL_ELEMENTS refers to the number of elements in the data for each index, while n is a particular choice of internal element.
This is needed when the data has further structure than the index structure, such as for example when for each index the data has been calculated for several energies.
If the data has no internal structure, or fewer than three indices, the corresponding variables are removed.
For example, if the data is two-dimensional and has no internal structure the data is accessed as
```cpp
	data[ranges[1]*x + y];
```

We finally note that while the Ranges format retains structural information about the problem, it does not retain the actual Index structure.
That is, although the x, y, and z variables bear resemblance to the corresponding subindices in the original Index structure, they have no real relation to each other.
Therefore it is not possible to extract elements from a @link TBTK::Property::AbstractProperty Property@endlink on the Ranges format using the original @link Indices@endlink on the form {x, y, z, s}.

## Custom
The Custom format allows for @link TBTK::Property::AbstractProperty Properties@endlink without a particular grid structure to be extracted.
For example when some Property is extracted from a molecule or from a few points on a grid without any particular relation to each other.
However, while no grid structure is imposed on the Property, the Custom format has the benefit of preserving the Index structure.
What this means is that after a Property has been created, it is possible to request a particular element using the original Indices used to specify the @link Model@endlink.
The interface for doing so is through the function operator, which means that the Property can be seen as a function defined over the Indices for which it has been extracted.
To access a particular element of the Property, simply type
```cpp
	DataType &element = property({x, y, z, s}, n);
```
where DataType should be replaced with the particular data type for the Property, and the second argument should be ignored if the Property has no internal structure other than the @link Indices Index@endlink structure.

Some properties does not have the full @link Indices Index@endlink structure of the original problem.
For example may a @link TBTK::Property::AbstractProperty Property@endlink such as @link TBTK::Property::LDOS LDOS@endlink be calculated by summing over the spin subindex using the identifier IDX_SUM_ALL.
Other Properties may still have the full Index structure, but the Property may have one data element associated with a range of indices.
For example does the @link TBTK::Property::Magnetization Magnetization@endlink contain one @link TBTK::SpinMatrix SpinMatrix@endlink that contains information about both up and down spins at the same time.
A typical case like this occurs when IDX_SPIN has been inserted in one of the subindices of the full Index structure at extraction.
In these cases the s in {x, y, z, s} should be left unspecified, which is possible to do with help of the wildcard specifier that can be written as \_a\_ or IDX_ALL.
```cpp
	DataType &element = property({x, y, z, _a_});
```

By default the @link TBTK::Property::AbstractProperty Properties@endlink will generate an error if an @link Indices Index@endlink is supplied as argument for which the Property has not been extracted.
However, sometimes it is useful to override this behavior and make the Property instead return some default value (e.g. zero) when an otherwise illegal Index is supplied.
To do so, execute the following commands
```cpp
	property.setAllowIndexOutOfBoundsAccess(true);
	property.setDefaultValue(defaultValue);
```
We note that it is recommended to be cautious about turning this feature on, since out of bound access in most cases is a sign of a bug.
Such bugs will be immediately detected at execution if out of bounds access is turned off.

# Density {#Density}
The @link TBTK::Property::Density Density@endlink has DataType double and can be extracted on the Ranges or Custom format.
Assume an Index structure with two spatial subindices, one orbital subindex and one spin subindex {x, y, orbital, spin}, and that the orbital and spin subindices has been summed over using the IDX_SUM_ALL specifier at the point of extraction.
On Ranges format a specific element can the be accessed as
```cpp
	vector<int> ranges = density.getRanges();
	const vector<double> &data = density.getData();
	double &d = data[ranges[1]*x + y];
```
while on the Custom format it can be accessed as
```cpp
	double &d = density({x, y, _a_, _a_});
```

# DOS {#DOS}
The @link TBTK::Property::DOS DOS@endlink has DataType double and is a global @link TBTK::Property::AbstractProperty Property@endlink without @link Indices Index@endlink structure but with an energy variable.
The lower and upper bound for the energy variable and the number of energy points in the interval can be extracted as
```cpp
	double lowerBound = dos.getLowerBound();
	double upperBound = dos.getUpperBound();
	double resolution = dos.getResolution();
```
while an individual element can be extracted as
```cpp
	double &d = dos(n);
```
where 0 <= *n* < *resolution*

# EigenValues {#EigenValues}
The @link TBTK::Property::EigenValues EigenValues@endlink @link TBTK::Property::AbstractProperty Property@endlink has DataType double and is a global Property without @link Indices Index@endlink structure.
The number of eigenvalues can be extracted as
```cpp
	unsigned int numEigenValues = eigenValues.getSize();
```
while an individual eigen value can be extracted as
```cpp
	double &e = eigenValues(n);
```
where 0 <= *n* < *numEigenValues*.

# LDOS {#LDOS}
The @link TBTK::Property::LDOS LDOS@endlink has DataType double and can be extracted on the Ranges or Custom format.
Assume an @link Indices Index@endlink structure with two spatial subindices, one orbital subindex and one spin subindex {x, y, orbital, spin}, and that the orbital and spin subindices has been summed over using the IDX_SUM_ALL specifier at the point of extraction.
The lower and upper bound for the energy variable and the number of energy points in the interval can be extracted as
```cpp
	double lowerBound = ldos.getLowerBound();
	double upperBound = ldos.getUpperBound();
	double resolution = ldos.getResolution();
```
On Ranges format a specific element can the be accessed as
```cpp
	vector<int> ranges = ldos.getRanges();
	const vector<double> &data = ldos.getData();
	double &d = data[resolution*(ranges[1]*x + y) + n];
```
where 0 <= *n* < resolution, while on the Custom format it can be accessed as
```cpp
	double &d = ldos({x, y, _a_, _a_}, n);
```

# Magnetization {#Magnetization}
The @link TBTK::Property::Magnetization Magnetization@endlink has DataType @link TBTK::SpinMatrix SpinMatrix@endlink and can be extracted on the Ranges or Custom format.
Assume an @link Indices Index@endlink structure with two spatial subindices, one orbital subindex and one spin subindex {x, y, orbital, spin}.
Further assume that the orbital subindex has been summed over using the IDX_SUM_ALL specifier at the point of extraction, while the spin-index has been specified using the IDX_SPIN specifier.
On Ranges format a specific element can the be accessed as
```cpp
	vector<int> ranges = magnetiation.getRanges();
	const vector<SpinMatrix> &data = magnetization.getData();
	SpinMatrix &m = data[ranges[1]*x + y];
```
while on the Custom format it can be accessed as
```cpp
	SpinMatrix &m = magnetization({x, y, _a_, _a_});
```

# SpinPolarizedLDOS {#SpinPolarizedLDOS}
The @link TBTK::Property::SpinPolarizedLDOS SpinPolarizedLDOS@endlink has DataType SpinMatrix and can be extracted on the Ranges or Custom format.
Assume an @link Indices Index@endlink structure with two spatial subindices, one orbital subindex and one spin subindex {x, y, orbital, spin}.
Further assume that the orbital subindex has been summed over using the IDX_SUM_ALL specifier at the point of extraction, while the spin-index has been specified using the IDX_SPIN specifier.
The lower and upper bound for the energy variable and the number of energy points in the interval can be extracted as
```cpp
	double lowerBound = spinPolarizedLDOS.getLowerBound();
	double upperBound = spinPolarizedLDOS.getUpperBound();
	double resolution = spinPolarizedLDOS.getResolution();
```
On Ranges format a specific element can the be accessed as
```cpp
	vector<int> ranges = spinPolarizedLDOS.getRanges();
	const vector<SpinMatrix> &data = spinPolarizedLDOS.getData();
	SpinMatrix &m = data[resolution*(ranges[1]*x + y) + n];
```
where 0 <= *n* < *resolution*, while on the Custom format it can be accessed as
```cpp
	SpinMatrix &s = spinPolarizedLDOS({x, y, _a_, _a_}, n);
```

# WaveFunctions {#WaveFunctions}
The @link TBTK::Property::WaveFunctions WaveFunctions@endlink has DataType complex<double> and can be extracted on the Custom format.
Assume an @link Indices Index@endlink structure with two spatial subindices, one orbital subindex and one spin subindex {x, y, orbital, spin}.
The states for which WaveFunctions contains wave functions can be extracted as
```cpp
	vector<unsigned int> &states = waveFunction.getStates();
```
On the Custom format a specific element can be accessed as
```cpp
	complex<double> &w = waveFunctions({x, y, orbital, spin}, n);
```
where *n* is one of the numbers contained in *states*.

@link ImportingAndExportingData Next: Importing and exporting data@endlink
@page ImportingAndExportingData Importing and exporting data
#  External storage {#ExternalStorage}
While the classes described in the other Chapters allow data to be stored in RAM during execution, it is important to also be able to store data outside of program memory.
This allows for data to be stored in files in between executions, to be exported to other programs, for external input to be read in, etc.
TBTK therefore comes with two methods for writing data structures to file on a format that allows for them to later be read into the same data structures, as well as one method for reading parameter files.

The first method is in the form of a @link TBTK::FileWriter FileWriter@endlink and @link TBTK::FileReader FileReader@endlink class, which allows for @link Properties@endlink and @link Model Models@endlink to be written into HDF5 files.
The HDF5 file format (https://support.hdfgroup.org/HDF5/) is a file format specifically designed for scientific data and has wide support in many languages.
Data written to file using the FileWriter can therefore easily be imported into for example MATLAB or python code for post-processing.
This is particularly true for Properties stored on the Ranges format (see the Properties chapter), since the data sections in the HDF5 files will preserve the Ranges format.

Many classes in TBTK can also be serialized, which mean that they are turned into strings.
These strings can then be written to file or passed as arguments to the constructor for the corresponding class to recreate a copy of the original object.
TBTK also contains a class called @link TBTK::Resource Resource@endlink, which allows for very general input and output of strings, including reading data immediately from the web.
In combination these two techniques allows for very flexible export and import of data that essentially allows large parts of the current state of the program to be stored in permanent memory.
The goal is to make almost every class serializable.
This would essentially allow a program to be serialized in the middle of execution and restarted at a later time, or allow for truly distributed applications to communicate their current state across the Internet.
However, this is a future vision not yet fully reached.

Finally, TBTK also contains a @link TBTK::FileParser FileParser@endlink that can parse a structured parameter file and create a @link TBTK::ParameterSet ParameterSet@endlink.

# FileReader and FileWriter {#FileReaderAndFileWriter}
The HDF5 file format that is used for the @link TBTK::FileReader FileReader@endlink and @link TBTK::FileWriter FileWriter@endlink essentially implements a UNIX like file system inside a file for structured data.
It allows for arrays of data, together with meta data called attributes, to be stored in datasets inside the file that resembles files.
When reading and writing data using the FileReader and FileWriter, it is therefore common to write several objects into the same HDF5-file.
The first thing to know about the FileReader and FileWriter is therefore that the current file it is using is chosen by typing
```cpp
	FileReader::setFileName("Filename.h5");
```
and similar for the FileWriter.
It is important to note here that the FileReader and FileWriter acts as global state machines.
What this means is that whatever change that is made to them at runtime is reflected throught the code.
If this command is executed in some part of the code, and then some other part of the code is reading a file, it will use the file "Filename.h5" as input.
It is possible to check whether a particular file already exists by first setting the filename and the call
```cpp
	bool fileExists = FileReader::exists();
```
and similar for the FileWriter.

A second important thing to know about HDF5 is that, although it can write new datasets to an already existing file, it does not allow for data sets to be overwritten.
If a program is meant to be run repeatedly, overwriting the previous data in the file each time it is rerun, it is therefore required to first delete the previously generated file.
This can be done after having set the filename by typing
```cpp
	FileWriter::clear();
```
A similar call also exists for the @link TBTK::FileReader FileReader@endlink, but it may seem harder to find a logical reason for calling it on the FileReader.

A @link Model@endlink or @link Properties Property@endlink can be written to file as follows
```cpp
	FileWriter::writeDataType(dataType);
```
where *DataType* should be replaced by one of the DataTypes listed below, and *data* should be an object of this data type.
|Supported DataTypes|
|-------------------|
| Model             |
| EigenValues       |
| WaveFunction      |
| DOS               |
| Density           |
| Magnetization     |
| LDOS              |
| SpinPolarizedLDOS |
| ParameterSet      |

By default the @link TBTK::FileWriter FileWriter@endlink writes the data to a dataset with the same name as the DataType listed above.
However, sometimes it is useful to specify a custom name, especially if multiple data structures of the same type are going to be written to the same file.
It is therefore possible to pass a second parameter to the write function that will be used as name for the dataset
```cpp
	FileWriter::writeDataType(data, "CustomName");
```

The interface for reading data is completely analogous to that for writing and takes the form
```cpp
	DataType data = FileReader::readDataType();
```
where DataType once again is a placeholder for one of the actual data type names listed in the table above.

# Serializable and Resource
Serialization is a powerful technique whereby an object is able to convert itself into a string.
If some classes implements serialization, it is simple to write new serializable classes that consists of such classes since the new class can serialize itself by stringing together the serializations of its components.
TBTK is designed to allow for different serialization modes.
Some types of serialization may be simpler or more readable in case they are not meant to be imported back into TBTK, while others might be more efficient in terms of execution time and memory requirements.
However, currently only serialization into JSON is implemented to any significant extent.
We will therefore only describe this mode here.

If a class is serializable, which means it either inherits from the @link TBTK::Serializable Serializable@endlink class, or is pseudo-serializable by implementing the *serialize()* function, it is possible to create a serialization of a corresponding object as follows
```cpp
	string serialization
		= serializeabelObject.serialize(Serializable::Mode::JSON);
```
Currently the @link Model@endlink and all @link Properties@endlink can be serialized like this.
For clarity considering the @link Model@endlink class, a Model can be recreated from a serialization string as follows
```cpp
	Model model(serialization, Serializable::Mode::JSON);
```
The notation for recreating other types of objects is the same, with Model replaced by the class name of the object of interest.

Having a way to create serialization strings and to recreate objects from such strings, it is useful to also be able to simply write and read such strings to and from file.
For this TBTK provides a class called @link TBTK::Resource Resource@endlink.
The interface for writing a string to file using a resource is
```cpp
	Resource resource;
	resource.setData(someString);
	resource.write("Filename");
```
Similarly a string can be read from file using
```cpp
	resource.read("Filename");
	const string &someString = resource.getData();
```

The @link TBTK::Resource Resource@endlink is, however, more powerful than demonstrated so far since it in fact implements an interface for the cURL library (https://curl.haxx.se).
This means that it for example is possible to read input from a URL instead of from file.
For example, a simple two level system is available at http://www.second-quantization.com/ExampleModel.json that can be used to construct a @link Model@endlink as follows
```cpp
	resource.read("http://www.second-quantization.com/ExampleModel.json");
	Model model(resource.getData(), Serializable::Mode::JSON);
	model.construct();
```

# FileParser and ParameterSet {#FileParserAndParameterSet}
While the main purpose of the other two methods is to provide methods for importing and exporting data that faithfully preserve the data structures that are used internally by TBTK, it is also often useful to read other information from files.
In particular, it is useful to be able to pass parameter values to a program through a file, rather than to explicitly type the parameters into the code.
Especially since the later option requires the program to be recompiled every time a parameter is updated.

For this TBTK provides a @link TBTK::FileParser FileParser@endlink and a @link TBTK::ParameterSet ParameterSet@endlink.
In particular, together they allow for files formated as follows to be read
<pre>
	int     sizeX       = 50
	int     sizeY       = 50
	double  radius      = 10
	complex phaseFactor = (1, 0)
	bool    useGPU      = true
	string  filename    = Model.json
</pre>
First the file can be converted into a ParameterSet as follows
```cpp
	ParameterSet parameterSet = FileParser::parseParameterSet("Filename");
```
Once the ParameterSet is created, the variables can be accessed
```cpp
	int sizeX                   = parameterSet.getInt("sizeX");
	int sizeY                   = parameterSet.getInt("sizeY");
	double radius               = parameterSet.getDouble("radius");
	complex<double> phaseFactor = parameterSet.getComplex("phaseFactor");
	bool useGPU                 = parameterSet.getBool("useGPU");
	string filename             = parameterSet.getString("filename");
```

@link Streams Next: Streams@endlink
@page Streams Streams
@link TBTK::Streams See more details about the Streams in the API@endlink

# Customizable Streams {#CustomizablStreams}
It is often useful to print information to the screen during execution.
Both for the sake of providing information about the progress of a calculation and for debuging code during development.
It is perfectly possible to use the standard C style *printf()* or C++ style *cout* streams for these purposes.
However, TBTK provides its own @link TBTK::Streams Stream@endlink interface that allows for customization of the output such as easy redirection of output to a logfile, etc.
Moreover, all TBTK functions use the Stream interface, and it is therefore useful to know how to handle these Streams in order to for example mute TBTK.

# Streams::out, Streams::log, and Streams::err {#OutLogAndErr}
The @link TBTK::Streams Stream@endlink interface has three different output channels called Streams::out, Streams::log, and Streams::err.
The Streams::out and Streams::err channels are by default equivalent to *cout* and *cerr* and is meant for standard output and error output, respectively.
In addition, the two buffers are forked to the Streams::log buffer which by default does nothing.
However, it is possible to make Streams::log write to an output file by typing
```cpp
	Streams::openLog("Logfile");
```
To ensure that all information is written to file at the end of a calculation, a corresponding close call should be made at the end of the program
```cpp
	Streams::closeLog();
```
It is further possible to turn of the output that is directed to *cout* as follows
```cpp
	Streams::setStdMuteOut();
```
while the output to *cerr* is muted by
```cpp
	Streams::setStdMuteErr();
```
We note that if a log is opened, muting any of these two channels will not turn of the output written to the logfile.
It is therefore possible to mute any output that otherwise would have gone to the screen and only redirect it to a file.
However, it is recommended to not mute the error stream, since *cerr* is designed to not be buffered, while the other streams are.
This means that information written to *cerr* before a crash will be guaranteed to reach the screen, while information written to the other streams do not provide this guarantee.

# Communicators {#Communicators}
Although not part of the actual @link TBTK::Streams Stream@endlink interface, many classes implements a so called @link TBTK::Communicator Communicator@endlink interface.
It is useful to know that in addition to muting the Streams themselves it is possible to globally mute all Communicators by typing
```cpp
	Communicator::setGlobalVerbose(false);
```
or individual objects implementing the Communicator interface using
```cpp
	communicator.setVerbose(false);
```

@link Timer Next: Timer@endlink
@page Timer Timer
@link TBTK::Timer See more details about the Timer in the API@endlink

# Profiling {#Profiling}
In a typical program most of the execution time is spent in a small fraction of the code.
It is therefore a good coding practice to first focus on writing a functional program and to then profile it to find eventual bottlenecks.
Optimization effort can then be spent on those parts of the code where it really matters.
Doing so allows for a high level of abstraction to be maintained, which reduces development time and makes the code more readable and thereby less error prone.
To help with profiling code, TBTK has a simple to use @link TBTK::Timer Timer@endlink class which can be used either as a timestamp stack or as a set of accumulators.
It is also possible to mix the two modes, using the timestamp stack for some measurements while simultaneously using the accumulators for other.

# Timestamp stack {#TimestampStack}
To time a section, all that is required is to enclose it between a *Timer::tick()* and a *Timer::tock()* call
```cpp
	Timer::tick("A custom tag");
	//Some code that is being timed.
	//...
	Timer::tock();
```
The tag string passed to *Timer::tick()* is optional, but is useful when multiple sections are timed since it will be printed together with the actual time when *Timer::tock()* is called.

When used as above, the @link TBTK::Timer Timer@endlink acts as a stack.
When a call is made to *Timer::tick()*, a new timestamp and tag is pushed onto the stack, and the *Timer::tock()* call pops the latest call and prints the corresponding time and tag.
It is therefore possible to nest Timer calls as follows
```cpp
	Timer::tick("Full loop");
	for(unsigned int m = 0; m < 10; m++){
		Timer::tock("Inner loop");
		for(unsigned int n = 0; n < 100; n++){
			//Do something
			//...
		}
		Timer::tock();
	}
	Timer::tock();
```
This will result in the Timer timing the inner loop ten times, each time printing the execution time together with the tag 'Inner loop'.
After the ten individual timing event have been completed, the Timer will also print the time it took for the full nested loop to execute together with the tag 'Full loop'.

# Accumulators {#Accumulators}
Sometimes it is useful to measure the accumulated time taken by one or several pieces of code that are not necessarily executed without other code being executed in between.
For example, consider the following loop
```cpp
	for(unsigned int n = 0; n < 1000000; n++){
		task1();
		task2();
	}
```
This piece of code may have been identified as a bottleneck in the program, but it is not clear which of the two tasks that is responsible for it.
Moreover, the time taken for each task may vary from call to call and therefore it is only useful to know the accumulated time taken for all 1,000,000 iterations.
For cases like this the @link TBTK::Timer Timer@endlink provides the possibility to create accumulators as follows
```cpp
	unsigned int accumulatorID1 = Timer::createAccumulator("Task 1");
	unsigned int accumulatorID2 = Timer::createAccumulator("Task 2");
```
The IDs returned by these functions can then be passed to the *Timer::tick()* and *Timer::tock()* calls to make it use these accumulators instead of the stack.
For the loop above we can now write
```cpp
	for(unsigned int n = 0; n < 1000000; n++){
		Timer::tick(accumulatorID1);
		task1();
		Timer::tock(accumulatorID1);

		Timer::tick(accumulatorID2);
		task2();
		Timer::tock(accumulatorID2);
	}	
```
Since the accumulators are meant to accumulate time over multiple calls, there is no reason for *Timer::tock()* to print the time each time it is called.
Instead the @link TBTK::Timer Timer@endlink has a special function for printing information about the accumulators, which will print the accumulated time and tags for all the currently created accumulators.
```cpp
	Timer::printAccumulators();
```

@link FourierTransform Next: FourierTransform@endlink
@page FourierTransform FourierTransform
@link TBTK::FourierTransform See more details about the FourierTransform in the API@endlink

# Fast Fourier transform {#FastFourierTransform}
One of the most commonly employed tools in physics is the Fourier transform and TBTK therefore provides a class that can carry out one-, two-, and three-dimensional Fourier transforms.
The class is a wrapper for the FFTW3 library (http://www.fftw.org), which implements an optimized version of the fast Fourier transform (FFT).

## Basic interface {#BasicInterface}
The basic interface for executing a transform is
```cpp
	FourierTransform::transform(in, out, SIZE_X, SIZE_Y, SIZE_Z, SIGN);
```
where the SIZE_Y and SIZE_Z can be dropped depending on the dimensionality of the transform.
Further, *in* and *out* are *complex<double>* arrays with SIZE_X*SIZE_Y*SIZE_Z elements, and SIGN should be -1 or 1 and determines the sign in the exponent of the transform.
The normalization factor is \f$\sqrt{SIZE\_X\times SIZE\_Y\times SIZE\_Z}\f$.

For simplicity the @link TBTK::FourierTransform FourierTransform@endlink also has functions with special names for the transforms with positive and negative sign.
The transform with negative sign can be called as
```cpp
	FourierTransform::forward(in, out, SIZE_X, SIZE_Y, SIZE_Z);
```
while the transform with positive sign can be called as
```cpp
	FourierTransform::inverse(in, out, SIZE_X, SIZE_Y, SIZE_Z);
```

## Advanced interface {#AdvancedInterface}
While the basic interface is very convenient, each such call involves a certain amount of overhead.
First, the FFTW3 library requires a plan to be setup prior to executing a transform, which involves a certain amount of computation ahead of the actual transform in order to figure out the optimal configuration.
For similar transforms it is possible to do such calculations once and reuse the same plan.
Second, by default the transform uses a specific normalization and it can be convenient to specify a custom normalization factor, or to set the normalization factor to 1, in which case the normalization step is avoided completely.
For this reason it is possible to setup a plan ahead of execution that both wraps the FFTW3 plan, as well as contains information about the normalization.

The creation of a plan mimics the interface for performing basic transforms
```cpp
	FourierTransform::Plan<complex<double>> plan(
		in,
		out,
		SIZE_X,
		SIZE_Y,
		SIZE_Z,
		SIGN
	);
```
Similar calls are available for creating forward and inverse transforms without explicitly specifying the sign and are called FourierTransform::ForwardPlan<complex<double>> and FourierTransform::InversePlan<complex<double>>, respectivley.
The normalization factor is set using
```cpp
	plan.setNormalizationFactor(1.0);
```

Once a plan is setup, repeated transforms can be carried out on the data in the *in* and *out* arrays by simply typing
```cpp
	FourierTransform::transform(plan);
```

@link Array Next: Array@endlink
@page Array Array
@link TBTK::Array See more details about the Array in the API@endlink

# Multi-dimensional arrays {#MultiDimensionalArrays}
One of the most common storage structures is the array.
TBTK therefore has a simple @link TBTK::Array Array@endlink class that allows for multi-dimensional data to be stored.
Such an Array can be created as follows
```cpp
	Array<DataType> array({SIZE_0, SIZE_1, SIZE_2});
```
where DataType should be replace by the specific data type of interest.
While the code above will create a three-dimensional array with dimensions (SIZE_0, SIZE_1, SIZE_2), it is possible to pass an arbitrary number of arguments to the constructor to create an Array of any dimension.

By default an @link TBTK::Array Array@endlink is uninitialized at creation, but it is possible to also supply a second argument at creation that will be used to initialize each element in the array.
For example, it is possible to initialize a three-dimensional array of doubles with zeros in the following way
```cpp
	Array<double> array({SIZE_0, SIZE_1, SIZE_2}, 0);
```
Once created it is possible to access the ranges of the array using
```cpp
	const vector<unsigned int> &ranges = array.getRanges();
```

An individual element in the @link TBTK::Array Array@endlink can be accessed using
```cpp
	DataType &data = array[{x, y, z}];
```
where 0 <= x < ranges[0], 0 <= y < ranges[1], and 0 <= z < ranges[2].

Given that the DataType supports the corresponding operators, it is also possible to add and subtract @link TBTK::Array Arrays@endlink from each other
```cpp
	Array<DataType> sum        = array0 + array1;
	Array<DataType> difference = array0 - array1;
```
as well as multiply and divide them by an *element* of the given DataType
```cpp
	Array<DataType> product  = element*array;
	Array<DataType> quotient = array/element;
```

A subset of an @link TBTK::Array Array@endlink can also be extracted using
```cpp
	Array<DataType> array2D = array.getSlice({x, _a_, _a_});
```
which in this case will extract the two-dimensional slice of the Array for which the first index is 'x'.
The new Array is a pure two-dimensional Array from which elements can be extracted using
```cpp
	DataType &element = array2D[{y, z}];
```

@link Plotting Next: Plotting@endlink
@page Plotting Plotting
@link TBTK::Plot::Plotter See more details about the Plotter in the API@endlink

# Quick and dirty {#QuickAndDirty}
The final step in most calculations involve plotting the results and TBTK therefore have limited support for plotting.
The current plotting abilities are restricted and rough, and currently the user is therefore recommended to use some external tool for final production plots.
However, the currently available plotting tools can be handy for quick and dirty plotting.
Especially, it allows for visualization to occur even in the middle of a calculation, which can be particularly useful during development.

# Plotter {#Plotter}
Plotting immediately from C++ code can be done using the @link TBTK::Plot::Plotter Plotter@endlink class and a Plotter is created as
```cpp
	Plotter plotter;
```
The Plotter currently generates pixel graphics and the width and height of the canvas can be set using
```cpp
	plotter.setWidth(WIDTH);
	plotter.setHeight(HEIGHT);
```
where *WIDTH* and *HEIGHT* are positive integers.
In order to save a plot type
```cpp
	plotter.save("FigureName.png");
```
Here it is important to make sure the filename ends with an image format such as '.png' for the call to succeed, as it will be used by the underlying OpenCV library to determine the file format of the resulting file.
For an up to date list of the supported file formats the reader is referred to the OpenCV documentation (https://opencv.org/).

It is possible to plot multiple data sets in the same graph, which can be done by setting *hold* to true
```cpp
	plotter.setHold(true);
```
When *hold* is set to true, it is also useful to be able to clear the plot, which is done as follows
```cpp
	plotter.clear();
```

By default the axes of a plot are automatically scaled to the bounds of the data.
However, it is also possible to specify the bounds using
```cpp
	plotter.setBoundsX(-1, 1);
	plotter.setBoundsY(0, 10);
```
or equivalently through a single call using
```cpp
	plotter.setBounds(-1, 1, 0, 10);
```
To return to auto scaling after bounds have been specified, use
```cpp
	plotter.setAutoScaleX(true);
	plotter.setAutoScaleY(true);
```
or simultaneously turn on auto scaling for both axes using
```cpp
	plotter.setAutoScale(true);
```

## Decoration
Many of the plot routines also accept a Decoration object as last argument to specify line the color and line style used for the data.
A typical plot command with a Decoration object look like
```cpp
	plotter.plot(
		data,
		Decoration(
			{192, 64, 64},
			Decoration::LineStyle::Line
		)
	);
```
Here the first argument to the Decoration object is the color in RGB format and each entry can be a value between 0 and 255.
The second argument is the line style, and currently this argument has to be supplied independently of whether the data actually can be plotted as lines or not.
The possible values are Decoration::LineStyle::Line and Decoration::LineStyle::Point.

## Plotting individual data points
Individual data points can be plotted using
```cpp
	plotter.plot(x, y);
```

## Plotting 1D data
One-dimensional data can be plotted using
```cpp
	plotter.plot(data);
```
where *data* either is an *std::vector* or a one-dimensional @link Array@endlink.
By default the x-axis will start at 0 and increment 1 per data point, but it is also possible to specify the x-values for the data points by instead using
```cpp
	plotter.plot(axis, data);
```
where *axis* is of the same type and size as *data*.

## Plotting 2D data
Two-dimensional plots can be plotted using
```cpp
	plotter.plot(data);
```
where *data* either is of type *std::vector<std::vector<double>>* or a two-dimensional @link Array@endlink.

# Plotting scripts {#PlottingScripts}
TBTK also have prepared plotting scripts written in python that can be used to plot @link Properties@endlink that has been saved to file using the @link TBTK::FileWriter FileWriter@endlink.
For these plotting scripts to work, the Properties has to be extracted on the Ranges (or None) format (see the PropertyExtractor and Properties chapters).
Some quantities can also only be plotted if they have been extracted for some particular dimensionality.
The plotting scripts can be run immediately from the terminal using the syntax
```bash
	TBTKPlotQuantity.py File.h5 parameters
```
where *Quantity* is a placeholder for the name of the relevant quantity, *File.h5* is the HDF5 file to which the Property has been written, and *parameters* is zero or more parameters needed to plot the data.
These scripts are limited to a few special usage cases due to the fact that different dimensionalities require widely different types of plots to be made.
However, they can be used in these specific cases, or the plotting scripts which are available in the folder TBTK/Visualization/python can be used as templates for writing customized plotting scripts.

## Density
The @link TBTK::Property::Density Density@endlink can be plotted if it has been extracted on a two-dimensional grid using
```bash
	TBTKPlotDensity.py File.h5
```

## DOS
The @link TBTK::Property::DOS DOS@endlink can be plotted using
```bash
	TBTKPlotDOS.py File.h5 sigma
```
where *sigma* is a decimal number specifying the amount of Gaussian smoothing that will be applied along the energy axis.

## EigenValues
The @link TBTK::Property::EigenValues EigenValues@endlink can be plotted as a monotonously increasing line using
```bash
	TBTKPlotEigenValues.py File.h5
```

## LDOS
The @link TBTK::Property::LDOS LDOS@endlink can be plotted if it has been extracted along a one-dimensional line using
```bash
	TBTKPlotLDOS.py File.h5 sigma
```
where *sigma* is a decimal number specifying the amount of Gaussian smoothing that will be applied along the energy axis.

## Magnetization
The @link TBTK::Property::Magnetization Magnetization@endlink can be plotted if it has been extracted on a two-dimensional grid using
```bash
	TBTKPlotMagnetization.py File.h5 theta phi
```
where *theta* and *phi* are the polar and azimuthal angles, respectively, of the spin-polarization axis of interest.

## SpinPolarizedLDOS
The @link TBTK::Property::SpinPolarizedLDOS SpinPolarizedLDOS@endlink can be plotted if it has been extracted along a one-dimensional line using
```bash
	TBTKPlotSpinPolarizedLDOS.py File.h5 theta phi sigma
```
where *theta* and *phi* are the polar and azimuthal angles, respectively, of the spin-polarization axis of interest.
Further, *sigma* is a decimal number specifying the amount of Gaussian smoothing that will be applied along the energy axis.

