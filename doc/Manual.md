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

# Solver::ChebyshevExpander {#SolverChebyshevExpander}
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

# A uniform interface for Solvers {#AUniformInterfaceForSolvers}
A @link TBTK::PropertyExtractor::PropertyExtractor PropertyExtractor@endlink provides an interfaces to a @link Solvers Solver@endlink through which physical @link Properties@endlink can be extracted.
Its purpose is to insulate the application developer from method specific details.
By wraping a Solver in a PropertyExtractor, it aquires an interface that is largely uniform across different Solvers.
This makes it possible to change Solver without having to modify the code used to extract Properties.

Each @link Solvers Solver@endlink comes with its own @link TBTK::PropertyExtractor::PropertyExtractor PropertyExtractor@endlink.
The PropertyExtractors have the same name as the corresponding Solver, but exists in the PropertyExtractor namespace rather than the Solver namespace.
For example, @link TBTK::Solver::Diagonalizer Solver::Diagonalizer@endlink and @link TBTK::PropertyExtractor::Diagonalizer PropertyExtractor::Diagonalizer@endlink.
It is created as follows.
```cpp
	PropertyExtractor::Diagonalizer propertyExtractor(solver);
```

The PropertyExtractors corresponding to the other three production ready Solvers are:
- @link TBTK::PropertyExtractor::BlockDiagonalizer PropertyExtractor::BlockDiagonalizer@endlink
- @link TBTK::PropertyExtractor::ArnoldiIterator PropertyExtractor::ArnoldiIterator@endlink
- @link TBTK::PropertyExtractor::ChebyshevExpander PropertyExtractor::ChebyshevExpander@endlink.

Since not every @link Solvers Solver@endlink can be used to calculate every @link Properties Property@endlink, the @link TBTK::PropertyExtractor::PropertyExtractor PropertyExtractors@endlink are only approximately uniform.
For many standard Properties the PropertyExtractors therefore fall back on printing an error message whenever a given Property cannot be calculated.
This informs the application developer that another Solver may be required for the given task.

# Extracting Properties {#ExtractingProperties}
The Properties that are extracted with the @link TBTK::PropertyExtractor::PropertyExtractor PropertyExtractors@endlink can be stored on different formats: None, Ranges, and Custom.
This refers to how the @link Indices@endlink associated with the Properties are handled internally (more details can be found in the @link Properties@endlink chapter).
Which format the Property is extracted on depends on the way the PropertyExtractor is called.

@link Properties@endlink without an @link Indices Index@endlink structure have the format None and are extracted through calls with zero arguments.
```cpp
	propertyExtractor.calculateProperty();
```
Note that *Property* in *calculateProperty()* is a placeholder for the name of the actual Property to exract.

@link Properties@endlink with an @link Indices Index@endlink structure can be extracted on the Ranges format by passing in a pattern-ranges pair.
```cpp
	propertyExtractor.calculateProperty(pattern, ranges);
```
Here pattern is an Index such as {3, IDX_X, IDX_SUM_ALL, IDX_Y}.
It tells the @link TBTK::PropertyExtractor::PropertyExtractor PropertyExtractor@endlink to extract the Property for all Indices of the form {3, *, *, *}, summing over the third Subindex.
Similarly, ranges is an Index such as {1, 10, 5, 2}.
It tells the PropertyExtractor that the second, third, and fourth Subindices runs over 0 to 9, 0 to 5, and 0 to 1, respectively.
The first Subindex is ignored, but is usefully set to 1 to calrify that it takes on a single value.

@link Properties@endlink with an @link Index@endlink structure can also be extracted on the Custom format by passing in a list of patterns.
```cpp
	propertyExtractor.calculateProperty({
		{0, _a_, IDX_SUM_ALL},
		{_a_, 5, IDX_SUM_ALL}
	});
```
This tells the @link TBTK::PropertyExtractor::PropertyExtractor PropertyExtractor@endlink to include all Indices of the form {0, *, *} and {*, 5, *}, summing over the third Subindex.
The list of patterns can be arbitrary long.
We note the extra pair of curly braces next to the parantheses.
Without these, the call would have been interpreted as an attempt to extract the Property on the Ranges format.

# Energy dependent Properties {#EnergyDependentProperties}
Many @link Properties@endlink are energy dependent.
It is possible to set the range and resolution of the energy interval for which the PropertyExtractor calculates them.
```cpp
	propertyExtractor.setEnergyWindow(
		lowerBound,
		upperBound,
		resolution
	);
```
The *resolution* refers to the number of points between *lowerBound* and *upperBound* that the Property is calculated for.

# Examples {#PropertyExtractorsExamples}

## DOS
The density of states (@link TBTK::Property::DOS DOS@endlink) has no @link Indices@endlink accoiated with it and can be extracted on the None format.
```cpp
	Property::DOS dos = propertyExtractor.calculateDOS();
```

## Density
Consider a @link Model@endlink with the @link Indices Index@endlink structure {x, y, z, spin}.
The @link TBTK::Property::Density Density@endlink in the plane \f$y = 10\f$ can be calculated on the Ranges format using
```cpp
	Property::Density density = propertyExtractor.calculateDensity(
		{IDX_X, 10, IDX_Y, IDX_SUM_ALL},
		{sizeX,  1, sizeZ,           2}
	);
```
Note that IDX_X and IDX_Y are not related to the @link TBTK::Subindex Subindices@endlink x and y.
Rather, they are labels indicating the first and second Subindex to loop over.

For a less regular set of @link Indices@endlink, we can use the Custom format instead.
The @link TBT::Property::Density Density@endlink in the plane \f$z = 10\f$, along the line \f$(y,z)=(5,15)\f$, and the spin down Density on the site \f$(x,y,z)=(0,0,0)\f$ can be extracted using.
```cpp
	Property::Density density = propertyExtractor.calculateDensity({
		{_a_, _a_, 10, IDX_SUM_ALL},
		{_a_,   5, 15, IDX_SUM_ALL},
		{  0,   0,  0,           1}
	});
```

## LDOS
Assuming the @link Indices Index@endlink structure {x, y, z, spin}, the @link TBTK::Property::LDOS LDOS@endlink can be extracted in the \f$z = 10\f$ plane and on the Ranges format using
```cpp
	Property::LDOS ldos = propertyExtractor.calculateLDOS(
		{ IDX_X,  IDX_Y,     10, IDX_SUM_ALL},
		{SIZE_X, SIZE_Y, SIZE_Z,           2}
	);
```

The Custom format can be used to extract it in the plane \f$z=10\f$, along the line \f$(y,z)=(5,15)\f$, and for the down spin on site \f$(x,y,z)=(0,0,0)\f$.
```cpp
	Property::LDOS ldos = propertyExtractor.calculateLDOS({
		{_a_, _a_, 10, IDX_SUM_ALL},
		{_a_,   5, 15, IDX_SUM_ALL},
		{  0,   0,  0,           1}
	});
```

## Magnetization
Assuming the @link Indices Index@endlink structure {x, y, z, spin}, the @link TBTK::Property::Magnetization Magnetization@endlink can be extracted in the \f$z = 10\f$ plane and on the Ranges format using
```cpp
	Property::Magnetization magnetization
		= propertyExtractor.calculateMagnetization(
			{ IDX_X,  IDX_Y,     10, IDX_SPIN},
			{SIZE_X, SIZE_Y, SIZE_Z,        2}
		);
```
Note the IDX_SPIN flag, which is necessary to indicate which @link TBTK::Subindex Subindex@endlink that corresponds to spin.

The Custom format can be used to extract it in the plane \f$z=10\f$, along the line \f$(y,z)=(5,15)\f$, and on the site \f$(x,y,z)=(0,0,0)\f$.
```cpp
	Property::Magnetization magnetization
		= propertyExtractor.calculateMagnetization({
			{_a_, _a_, 10, IDX_SPIN},
			{_a_,   5, 15, IDX_SPIN},
			{  0,   0,  0, IDX_SPIN}
		});
```

## SpinPolairzedLDOS
Assuming the @link Indices Index@endlink structure {x, y, z, spin}, the @link TBTK::Property::SpinPolarizedLDOS SpinPolarizedLDOS@endlink can be extracted in the \f$z = 10\f$ plane and on the Ranges format using
```cpp
	Property::SpinPolarizedLDOS spinPolarizedLDOS
		= propertyExtractor.calculateSpinPolarizedLDOS(
			{ IDX_X,  IDX_Y,     10, IDX_SPIN},
			{SIZE_X, SIZE_Y, SIZE_Z,        2}
		);
```
Note the IDX_SPIN flag, which is necessary to indicate which @link TBTK::Subindex Subindex@endlink that corresponds to spin.

The Custom format can be used to extract it in the plane \f$z=10\f$, along the line \f$(y,z)=(5,15)\f$, and on the site \f$(x,y,z)=(0,0,0)\f$.
```cpp
	Property::SpinPolarizedLDOS spinPolarizedLDOS
		= propertyExtractor.calculateSpinPolarizedLDOS({
			{_a_, _a_, 10, IDX_SPIN},
			{_a_,   5, 15, IDX_SPIN},
			{  0,   0,  0, IDX_SPIN}
		});
```

@link Properties Next: Properties@endlink
@page Properties Properties
@link TBTK::Property::AbstractProperty See more details about the Properties in the API@endlink

# Containers of physical Properties {#ContainerOfPhysicalProperties}
@link TBTK::Property::AbstractProperty Properties@endlink are containers of physical properties that can be calculated from a @link Model@endlink.
They store both the data itself and meta data such as the energy interval they have been calculated over.
Properties can also be used as a functions of the arguments for which they have been calculated.
This makes them convenient building blocks for applications that extends the capabilities beyond those of the standard @link Solvers@endlink.

This chapter discuss the Properties themselves.
For information on how to calculate them, see the @link PropertyExtractors@endlink chapter.

# EnergyResolvedProperties {#EnergyResolvedProperties}
Many @link TBTK::Property::AbstractProperty Properties@endlink are @link TBTK::Property::EnergyResolvedProperty EnergyResolvedProperties@endlink, which means that they contain data over some energy interval.
It is possible to get information about the interval as follows.
```cpp
	double lowerBound = energyResolvedProperty.getLowerBound();
	double upperBound = energyResolvedProperty.getUpperBound();
	unsigned resolution = energyResolvedProperty.getResolution();
```
Here *resolution* is the number of points with which the data is resolved between the *lowerBound* and the *upperBound*.

# Storage modes {#StorageModes}
There are three different storage modes for @link TBTK::Property::AbstractProperty Properties@endlink called None, Ranges, and Custom.
Here we describe the difference between these modes.
We also describe how the mode affects the way in which data can be accessed from the Property.

Independently of the storage mode, the total number of data points contained in the Property can be retrieved using
```cpp
	unsigned int size = property.getSize();
```

## None
This storage mode is for @link TBTK::Property::AbstractProperty Properties@endlink that have no @link Indices@endlink associated with them.
For example, the density of states (@link TBTK::Property::DOS DOS@endlink) and the @link TBTK::Property::EigenValues EigenValues@endlink.

If the Property is a function of energy, it is possible to get the nth element using the notation
```cpp
	property(n);
```

## Ranges
This storage mode is meant for extracting @link TBTK::Property::AbstractProperty Properties@endlink on a regular grid, e.g. the @link TBTK::Property::Density Density@endlink on a quadratic lattice.
It can be particularly useful when the Property is to be immediately exported for external postprocessing.

When stored on this format, the data and its ranges can be extracted as follows.
```cpp
	const vector<DataType> &data = property.getData();
	vector<int> ranges = property.getRanges();
```
Here DataType should be replaced by the particular data type of the Property's values.

If the data also is a function of energy, with resolution ENERGY_RESOLUTION, then the layout is such that the data can be accessed as follows.
```cpp
	data[ENERGY_RESOLUTION*(ranges[2]*(ranges[1]*x + y) + z) + n];
```
If the data instead would be two-dimensional and without energy dependence, the data is accessed as follows.
```cpp
	data[ranges[1]*x + y];
```

## Custom
This storage format is meant for @link TBTK::Property::AbstractProperty Properties@endlink that have been calculated for arbitrary @link Indices@endlink.
If a the Property has been calculated for a given Index, say {x, y, z, spin}, it is possible to access the value as follows.
```cpp
	property({x, y, z, spin});
```
If the Property is energy dependent, we instead use
```cpp
	property({x, y, z, spin}, n);
```
where n is an energy index.

### Properties with subindex specifiers
It is possible that a @link TBTK::Property::AbstractProperty Property@endlink has been calculated using a specifier in one of the @link TBTK::Subindex Subindices@endlink.
Consider for example the @link Indices Index@endlink structure {x, y, z, spin}, and the @link TBTK::Property::Density Density@endlink obtained through
```cpp
	Property::Density density = propertyExtractor.calculateDensity({
		{5, _a_, _a_, IDX_SUM_ALL}
	});
```
Here the Density is being calculated for all possible coordinates of the form \f$(5, y, z)\f$.
However, the subindex specifier IDX_SUM_ALL ensures the spin Subindex is summed over.
The resulting Density retains the original Index structure, including the spin Subindex.
To get the value of the Density at a specific point \f$(5, y, z)\f$, we call the Density with a wildcard in the spin Subindex.
```cpp
	density({5, y, z, _a_});
```

### Index out of bounds access
By default, a @link TBTK::Property::AbstractProperty Property@endlink will generate an error when trying to access a value that is not contained in the Property.
However, it is possible to configure the Property such that it instead return some default value (e.g. zero).
This can be done as follows.
```cpp
	property.setAllowIndexOutOfBoundsAccess(true);
	property.setDefaultValue(defaultValue);
```
We note that it is recommended to be cautious about turning this feature on.
Out of bounds access is often a sign that something is wrong and turning this on can therefore mask bugs.

# Examples {#PropertiesExamples}
See the @link PropertyExtractors@endlink chapter for details about the different way that @link TBTK::Property::AbstractProperty Properties@endlink can be extracted.

## Density {#Density}
Assume the Index structure {x, y, orbital, spin}.

<b>Ranges format</b>
```cpp
	Property::Density density = propertyExtractor.calculateDensity(
		{  _a_,   _a_, IDX_SUM_ALL, IDX_SUM_ALL},
		{sizeX, sizeY, numOrbitals,           2}
	);

	vector<int> ranges = density.getRanges();
	const vector<double> &data = density.getData();
	double d = data[ranges[1]*x + y];
```

<b>Custom format</b>
```cpp
	Property::Density density = propertyExtractor.calculateDensity({
		{_a_, _a_, IDX_SUM_ALL, IDX_SUM_ALL}
	});
	double d = density({x, y, _a_, _a_});
```

@link TBTK::Property::Density See more details about the Density in the API@endlink

## DOS {#DOS}
<b>None format</b>
```cpp
	Property::DOS dos = propertyExtractor.calculateDOS();

	double lowerBound = dos.getLowerBound();
	double upperBound = dos.getUpperBound();
	double resolution = dos.getResolution();

	double d = dos(n);
```
Here 0 <= *n* < *resolution*.

@link TBTK::Property::DOS See more details about the DOS in the API@endlink

## EigenValues {#EigenValues}
<b>None format</b>
```cpp
	unsigned int numEigenValues = eigenValues.getSize();
	double e = eigenValues(n);
```
Here 0 <= *n* < *numEigenValues*.

@link TBTK::Property::EigenValues See more details about the EigenValues in the API@endlink

## LDOS {#LDOS}
Assume the @link Indices Index@endlink structure {x, y, z, spin}.

<b>Ranges format</b>
```cpp
	Property::LDOS ldos = propertyExtractor.calculateLDOS(
		{  _a_,   _a_, IDX_SUM_ALL, IDX_SUM_ALL},
		{sizeX, sizeY, numOrbitals,           2}
	);

	double lowerBound = ldos.getLowerBound();
	double upperBound = ldos.getUpperBound();
	double resolution = ldos.getResolution();

	vector<int> ranges = ldos.getRanges();
	const vector<double> &data = ldos.getData();
	double d = data[resolution*(ranges[1]*x + y) + n];
```
Here 0 <= *n* < resolution.

<b>Custom format</b>
```cpp
	Property::LDOS ldos = propertyExtractor.calculateLDOS({
		{_a_, _a_, IDX_SUM_ALL, IDX_SUM_ALL}
	});

	double lowerBound = ldos.getLowerBound();
	double upperBound = ldos.getUpperBound();
	double resolution = ldos.getResolution();

	double d = ldos({x, y, _a_, _a_}, n);
```
Here 0 <= *n* < resolution.

@link TBTK::Property::LDOS See more details about the LDOS in the API@endlink

## Magnetization {#Magnetization}
Assume the @link Indices Index@endlink structure {x, y, orbital, spin}.

<b>Ranges format</b>
```cpp
	Property::Magnetization magnetization
		= propertyExtractor.calculateMagnetization(
			{  _a_,   _a_, IDX_SUM_ALL, IDX_SPIN},
			{sizeX, sizeY, numOrbitals,        2}
		);

	vector<int> ranges = magnetiation.getRanges();
	const vector<SpinMatrix> &data = magnetization.getData();
	SpinMatrix spinMatrix = data[ranges[1]*x + y];
```

<b>Custom format</b>
```cpp
	Property::Magnetization magnetization
		= propertyExtractor.calculateMagnetization({
			{_a_, _a_, IDX_SUM_ALL, IDX_SPIN}
		});

	SpinMatrix spinMatrix = magnetization({x, y, _a_, _a_});
```

@link TBTK::Property::Magnetization See more details about the Magnetization in the API@endlink<br />
@link TBTK::SpinMatrix See more details about the SpinMatrix in the API@endlink

## SpinPolarizedLDOS {#SpinPolarizedLDOS}
Assume the @link Indices Index@endlink structure {x, y, orbital, spin}.

<b>Ranges format</b>
```cpp
	Property::SpinPolarizedLDOS spinPolarizedLDOS
		= propertyExtractor.calculateSpinPolarizedLDOS(
			{  _a_,   _a_, IDX_SUM_ALL, IDX_SPIN},
			{sizeX, sizeY, numOrbitals,        2}
		);

	double lowerBound = spinPolarizedLDOS.getLowerBound();
	double upperBound = spinPolarizedLDOS.getUpperBound();
	double resolution = spinPolarizedLDOS.getResolution();

	vector<int> ranges = spinPolarizedLDOS.getRanges();
	const vector<SpinMatrix> &data = spinPolarizedLDOS.getData();
	SpinMatrix spinMatrix = data[resolution*(ranges[1]*x + y) + n];
```
Here 0 <= *n* < *resolution*.

<b>Custom format</b>
```cpp
	Property::SpinPolarizedLDOS spinPolarizedLDOS
		= propertyExtractor.calculateSpinPolarizedLDOS({
			{_a_, _a_, IDX_SUM_ALL, IDX_SPIN}
		});

	double lowerBound = spinPolarizedLDOS.getLowerBound();
	double upperBound = spinPolarizedLDOS.getUpperBound();
	double resolution = spinPolarizedLDOS.getResolution();

	SpinMatrix spinMatrix = spinPolarizedLDOS({x, y, _a_, _a_}, n);
```
Here 0 <= *n* < *resolution*.

@link TBTK::Property::SpinPolarizedLDOS See more details about the SpinPolarizedLDOS in the API@endlink<br />
@link TBTK::SpinMatrix See more details about the SpinMatrix in the API@endlink

## WaveFunctions {#WaveFunctions}
Assume the @link Indices Index@endlink structure {x, y, orbital, spin}.

<b>Custom format</b><br />
The @link TBTK::Property::WaveFunctions WaveFunctions@endlink are extracted somewhat differently from all other @link TBTK::Property::AbstractProperty Properties@endlink.
The second argument to the @link PropertyExtractors PropertyExtractor@endlink call is a list of the eigenstate indices to extract the WaveFunctions for.
This can also be set to \_a\_ to extract all states.
```cpp
	Property::WaveFunctions waveFunctions
		= propertyExtractor.calculateWaveFunctions(
			{{_a_, _a_, _a_, _a_}},
			{1, 3, 7}
		);

	vector<unsigned int> &states = waveFunction.getStates();
	complex<double> &w = waveFunctions({x, y, orbital, spin}, n);
```
where *n* is one of the numbers contained in *states*.

@link TBTK::Property::WaveFunctions See more details about the WaveFunctions in the API@endlink<br />

@link ImportingAndExportingData Next: Importing and exporting data@endlink
@page ImportingAndExportingData Importing and exporting data

#  External memory {#ExternalMemory}
TBTK facilitates the reading and writing of data to external memory through a few different scehems, which we describe here.

# FileParser and ParameterSet {#FileParserAndParameterSet}
The @link TBTK::FileParser FileParser@endlink can generate a @link TBTK::ParameterSet ParameterSet@endlink by parsing a file formated as follows.
```bash
	int     sizeX       = 50
	int     sizeY       = 50
	double  radius      = 10
	complex phaseFactor = (1, 0)
	bool    useGPU      = true
	string  filename    = Model.json
```
Assume the file is called "ParameterFile".
It is then possible to parse it and extract the values using
```cpp
	ParameterSet parameterSet
		= FileParser::parseParameterSet("ParameterFile");

	int sizeX                   = parameterSet.getInt("sizeX");
	int sizeY                   = parameterSet.getInt("sizeY");
	double radius               = parameterSet.getDouble("radius");
	complex<double> phaseFactor = parameterSet.getComplex("phaseFactor");
	bool useGPU                 = parameterSet.getBool("useGPU");
	string filename             = parameterSet.getString("filename");
```

# Serializable and Resource
Many classes implement the @link TBTK::Serializable Serializable@endlink interface.
This means that they can be converted into text strings that can then be converted back into objects.

For example, the following code first serializes a @link Model@endlink and then recreates a copy of it from the resulting serialization string.
```cpp
	string serialization = model.serialize(Serializable::Mode::JSON);
	Model copyOfModel(serialization, Serializable::Mode::JSON);
```
The parameter Serializable::Mode::JSON indicates that the serialization should be done to and from JSON.
Currently, this is the only widely supported serialization format in TBTK.

The power of serialization is that a string can be stored or sent anywhere.
The @link TBTK::Resource Resource@endlink class is meant to simplify this task.
For example, the serialization created above can be saved to a file called Model.json using
```cpp
	Resource resource;
	resource.setData(serialization);
	resource.write("Model.json");
```
The @link Model@endlink can then be read in and reconstructed in a different program using
```cpp
	Resource resource;
	resource.read("Model.json");
	Model model(resource.getData(), Serializable::Mode::JSON);
```

The @link TBTK::Resource Resource@endlink can also download data from a URL.
For example, the following code creates a @link Model@endlink from a file downloaded from www.second-quantization.com.
```cpp
	Resource resource;
	resource.read(
		"http://www.second-quantization.com/v2/ExampleModel.json"
	);
	Model model(resource.getData(), Serializable::Mode::JSON);
```

To be able to use the @link TBTK::Resource Resource@endlink class, [cURL](https://curl.haxx.se) must be installed when TBTK is compiled.

# FileReader and FileWriter {#FileReaderAndFileWriter}
The @link TBTK::FileReader FileReader@endlink and @link TBTK::FileWriter FileWriter@endlink can import and export @link Properties@endlink stored on the Ranges format.
They use the [HDF5](https://support.hdfgroup.org/HDF5/) file format, which is particularly suited for data with an array like structure.
HDF5 also has wide support in languages such as python, MATLAB, Mathematica, etc., which makes it easy to export Properties on the Ranges format to other languages.

Multiple @link Properties@endlink can be stored in the same HDF5 file.
To set the filename to read from and check whether the file exists, we can use the following code.
```cpp
	FileReader::setFileName("Filename.h5");
	bool fileExists = FileReader::exists();
```
The same code works with @link TBTK::FileReader FileReader@endlink interchanged with @link TBTK::FileWriter FileWriter@endlink.

If a @link Properties Property@endlink with a given name has already been written to the HDF5 file, it is not possible to overwrite it.
This is a limitation of the HDF5 file format itself.
Instead, the whole file (including all Properties that have been written to it) has to be deleted.
This is achieved through
```cpp
	FileWriter::clear();
```

With *DataType* replaced by a @link Properties Property@endlink name, a Property can be written to the currently selected HDF5 file using
```cpp
	FileWriter::writeDataType(property);
```
The possible Properties are the following
|Supported DataTypes|
|-------------------|
| EigenValues       |
| WaveFunctions     |
| DOS               |
| Density           |
| Magnetization     |
| LDOS              |
| SpinPolarizedLDOS |

By default, the FileWriter will write the @link Properties Property@endlink to a dataset with the same name as the Property.
However, it is possible to choose a custom name for the dataset using
```cpp
	FileWriter::writeDataType(data, "CustomName");
```

It is similarly possible to read a @link Properties Property@endlink using the call
```cpp
	Property::DataType property = FileReader::readDataType();
```
Here *DataType* is to be replaced by the particular name of the Property wanted.
To read a Property from a dataset set with a custom name, instead use
```cpp
	Property::DataType property = FileReader::readDataType("CustomName");
```

@link Streams Next: Streams@endlink
@page Streams Streams
@link TBTK::Streams See more details about the Streams in the API@endlink

# Customizable Streams {#CustomizablStreams}
TBTK provides a number of output @link TBTK::Streams Streams@endlink that can be used to print diagnostic information etc.
It is of course possible to use the C style *printf()* or C++ style *std::cout* for this.
However, the TBTK Streams can be customized in ways that are useful for numerical calculations.
Further, all information printed by TBTK is printed through these Streams.
So even if *printf()* or *std::out* is used in application code, knowledge about these Streams is useful for tweaking the output of TBTK itself.

# Streams::out, Streams::log, and Streams::err {#OutLogAndErr}
The @link TBTK::Streams Streams@endlink interface has three output channels called Streams::out, Streams::log, and Streams::err.
By default, Streams::out and Streams::err are maped directly onto *std::cout* and *std::cerr*, respectively.
These two Streams are also forked to Streams::log, which does nothing by default.
What this means is that by default Streams::out and Streams::err works just like *std::cout* and *std::cerr*.

## Opening and closing the log
It is possible to make Streams::log write to an output file by typing
```cpp
	Streams::openLog("Logfile");
```
In addition to directing the output of both Streams::out and Streams::err to the log file, this call will also write some diagnostic information.
This includes a timestamp and the version number and git hash of the currently installed version of TBTK.
This is information that ensures that the results can be reproduced in the future.
Knowing the version number makes it possible to recompile an application in the future using the exact same version of TBTK.
The git hash is there to make this possible even if a named release is not used.

A corresponding close call should be made before the application finishes to avoid losing any of the output.
```cpp
	Streams::closeLog();
```
This will also write a timestamp to the log before closing it.

## Muting the output to std::cout and std::cerr
It is also possible to mute the output that is directed to *std::cout* and *std::err* as follows
```cpp
	Streams::setStdMuteOut();
	Streams::setStdMuteErr();
```
This will not mute the output of Streams::out and Streams::err to the log.
If the log is opened, the result is therefore that the output is directed to the log file alone.
If the application code writes immediately to *std::cout*, this can be used to direct application specific output to the terminal and TBTK output to the log file.
We note, however, that it is recommended to not mute the output to cerr.
Doing so can result in some error messages being lost at a crash.

# Communicators {#Communicators}
Although not part of the @link TBTK::Streams Streams@endlink interface, many classes implements a @link TBTK::Communicator Communicator@endlink interface.
Instead of muting the Streams, it is possible to mute all such Communicators using
```cpp
	Communicator::setGlobalVerbose(false);
```
It is also possible to mute individual classes that implements the Communicator interface.
```cpp
	communicator.setVerbose(false);
```

In contrast to muting the Streams, as described above, this will mute the component completely.
A muted Communicator will therefore not even write to the log.

@link Timer Next: Timer@endlink
@page Timer Timer
@link TBTK::Timer See more details about the Timer in the API@endlink

# Profiling {#Profiling}
In a typical program, most of the execution time is spent in a small fraction of the code.
It is thefore useful to be able to time different parts of the code to figure out where optimization may be needed.
The @link TBTK::Timer Timer@endlink helps with this.
It can either be used as a timestamp stack or as a set of accumulators, or both at the same time.

# Timestamp stack {#TimestampStack}
To time a section, all that is required is to enclose it between a *Timer::tick()* and a *Timer::tock()* call.
```cpp
	Timer::tick("A custom tag");
	//Some code that is being timed.
	//...
	Timer::tock();
```
The tag string passed to *Timer::tick()* is optional, but is useful for distinguish between different timed sections.
When *Timer::tock()* is called, the tag together with the time that elapsed since the tick call is printed.

More specifically, *Timer::tick()* pushes a timestamp and a tag onto a stack, while *Timer::tock()* pops one off from it.
It is therefore possible to nest @link TBTK::Timer Timer@endlink calls as follows.
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
This results in the inner loop being timed ten times, each time printing the execution time together with the tag 'Inner loop'.
The entry corresponding to 'Full loop' remains on the stack throughout execution of the two nested loops.
When the final call to *Timer::tock()* occurs, this entry is poped from the stack and the full execution time is printed with the tag 'Full loop'.

# Accumulators {#Accumulators}
To understand the accumulators, assume that the following loop has been identified to be a bottleneck.
```cpp
	for(unsigned int n = 0; n < 1000000; n++){
		task1();
		task2();
	}
```
Next we want to figure out which of the two tasks that is responsible for the slow execution.
However, task 1 and 2 may have varying execution times.
Therefore, only the total time taken by each call over the 1,000,000 iterations is relevant.

In cases like these, we can use accumulators to time the code.
```cpp
	unsigned int accumulatorID1 = Timer::createAccumulator("Task 1");
	unsigned int accumulatorID2 = Timer::createAccumulator("Task 2");

	for(unsigned int n = 0; n < 1000000; n++){
		Timer::tick(accumulatorID1);
		task1();
		Timer::tock(accumulatorID1);

		Timer::tick(accumulatorID2);
		task2();
		Timer::tock(accumulatorID2);
	}	

	Timer::printAccumulators();
```
In the first two lines, we create two accumulators called "Task 1" and "Task 2" and get an ID for each of them.
We then pass these IDs to the tick and tock calls.
The time taken between such a pair of tick-tack calls are added to the accumulators with the given ID.
The final line prints a table displaying the total time accumulated in each accumulator.

@link FourierTransform Next: FourierTransform@endlink
@page FourierTransform FourierTransform
@link TBTK::FourierTransform See more details about the FourierTransform in the API@endlink

# Fast Fourier transform {#FastFourierTransform}
The @link TBTK::FourierTransform FourierTransform@endlink can calculate the one- two- and three-dimensional Fourier transform.
This is a wrapper class for the [FFTW3](http://www.fftw.org) library, which implements an optimized version of the fast Fourier transform (FFT).

## Basic interface {#BasicInterface}
The three-dimensional Fourier transform and inverse Fourier transform an be calculated using
```cpp
	FourierTransform::forward(in, out, SIZE_X, SIZE_Y, SIZE_Z);
	FourierTransform::inverse(in, out, SIZE_X, SIZE_Y, SIZE_Z);
```
Here *in* and *out* are c-arrays of type std::complex<double> with size \f$SIZE\_X\times SIZE\_Y\times SIZE\_Z\f$.
The normalization factor for each call is \f$\sqrt{SIZE\_X\times SIZE\_Y\times SIZE\_Z}\f$.
The one- and two-dimensional versions are obtained by droping the later arguments.

## Advanced interface {#AdvancedInterface}
When executing multiple similar transforms, it is possible to avoid some overhead by using the advanced interface.
This is done by first setting up a plan.
```cpp
	FourierTransform::ForwardPlan<complex<double>> plan(
		in,
		out,
		SIZE_X,
		SIZE_Y,
		SIZE_Z
	);
```
Plans for one- and two-dimensional transforms are obtained by droping the later arguments.
A corresponding plan for the inverse transform can be created by replacing ForwardPlan by InversePlan.

It is also possible to specify a custom normalization factor.
If the value is set to 1, the calculation will be avoided completely.
```cpp
	plan.setNormalizationFactor(1.0);
```
By refilling *in* with new data between each call, multiple transforms can now be calculated using
```cpp
	FourierTransform::transform(plan);
```

@link Array Next: Array@endlink
@page Array Array
@link TBTK::Array See more details about the Array in the API@endlink

# Multi-dimensional arrays {#MultiDimensionalArrays}
## Creating an Array
A three-dimensional @link TBTK::Array Array@endlink with the data type DataType can be created using
```cpp
	Array<DataType> array({SIZE_0, SIZE_1, SIZE_2});
```
The number of arguments in the list can be changed to get an Array of different dimensionality.

By default, the @link TBTK::Array Array@endlink is uninitialized after creation, but this can be changed by passing a second argument.
For example, an Array with type double can be initialized to zero as follows.
```cpp
	Array<double> array({SIZE_0, SIZE_1, SIZE_2}, 0);
```

## Getting the Array size and accessing its elements
The Array ranges and dimensionality can be retrieved using
```cpp
	const vector<unsigned int> &ranges = array.getRanges();
	unsigned int dimension = ranges.size();
```

It is also possible to access or assign values to individualt elements.
Assuming the that the data type is double, we can use
```cpp
	double data = array[{x, y, z}];
	array[{x, y, z}] = 1;
```
Here 0 <= x < ranges[0], 0 <= y < ranges[1], and 0 <= z < ranges[2].

## Array operations

If the DataType supports it, the following operations are possible.

<b>Addition</b>
```cpp
	Array<DataType> sum = array0 + array1;
```
<b>Subtraction</b>
```cpp
	Array<DataType> difference = array0 - array1;
```
<b>Multiplication</b>
```cpp
	Array<DataType> product = multiplier*array;
```
Here *multiplier* has the data type DataType.

<b>Division</b>
```cpp
	Array<DataType> quotient = array/divisor;
```
Here *divisor* has the the data type DataType.

## Slicing the Array
A lower-dimensional subset of an @link TBTK::Array Array@endlink can be extracted using
```cpp
	Array<DataType> array2D = array.getSlice({x, _a_, _a_});
```
This extracts the subset of *array* for which the first entry is 'x'.
The result is a two-dimensional Array from which we can access elements using
```cpp
	DataType element = array2D[{y, z}];
```

@link Plotting Next: Plotting@endlink
@page Plotting Plotting
@link TBTK::Plot::Plotter See more details about the Plotter in the API@endlink

# Internal and external plotting tools
TBTK provides methods for plotting both inside and outside the application.
However, we note that these tools are currently quite limited and other software is needed to produce high quality figures.
Nevertheless, the native plotting tools can be very useful during the development process.
They can also be very handy for getting quick insight into calculations while they are still running.

The internal plotting tool is the @link TBTK::Plotter Plotter@endlink, while the external tools consists of a number of python scripts.
Below, we describe each of these tools in more detail.

# Plotter {#Plotter}
The @link TBTK::Plot::Plotter Plotter@endlink can plot a number of data formats.
For example, data stored in *std::vector<double>*, @link Arrays Array@endlink, and some @link Properties@endlink.
It exists inside the Plot namespace and below we assume that the following line has been added at the top of the code.
```cpp
	using namespace Plot;
```

## Setting up and configuring a Plotter
We begin by listing some of the most common commands needed for setting up and configuring the @link TBTK::Plot::Plotter Plotter@endlink.

<b>Create a Plotter</b>
```cpp
	Plotter plotter;
```

<b>Setting the canvas size</b>
```cpp
	plotter.setWidth(width);
	plotter.setHeight(height);
```

<b>Hold data between successive plots</b><br />
By default, the Plotter clears the canvas between succesive calls.
This can be changed through the following call.
```cpp
	plotter.setHold(true);
```

<b>Clear the canvas</b>
```cpp
	plotter.clear();
```

<b>Save the canvas to file</b>
```cpp
	plotter.save("FigureName.png");
```

<b>Set axis bounds</b><br />
By default, the axes automatically scales to contain the data.
It is possible to fix the axis bounds with the following calls.
```cpp
	plotter.setBoundsX(-1, 1);
	plotter.setBoundsY(0, 10);
```
These calls can also be commbind into a single call.
```cpp
	plotter.setBounds(-1, 1, 0, 10);
```

<b>Reset to auto scaling axes.</b>
```cpp
	plotter.setAutoScaleX(true);
	plotter.setAutoScaleY(true);
```
These calls can also be combined into a single call.
```cpp
	plotter.setAutoScale(true);
```

<b>Decoration</b><br />
Many @link TBTK::Plotter Plotter@endlink routines accept a Decoration object to specify the color, line style, and line width/point size.
A typical plot command with a Decoration object look like
```cpp
	plotter.plot(
		data,
		Decoration(
			{192, 64, 64},
			Decoration::LineStyle::Line,
			size
		)
	);
```
Here the first argument to Decoration is the color in RGB format, with each entry a number between 0 and 255.
The second argument is the line style, while the third argument in the case of Decoration::LineStyle::Line is the line width.
If the line style instead is Decoration::LineStyle::Point, the third argument is the radius of the points.

## Examples
We here provide a few examples of how the @link TBTK::Plotter Plotter@endlink can be used.

<b>Individual data points</b>
```cpp
	plotter.plot(x, y);
```

<b>1D data</b><br />
If *data* is an *std::vector<double>* or a one-dimensional @link Arrays Array@endlink, it can be plotted using
```cpp
	plotter.plot(data);
```
By default, the x-axis will run from 0 to the number of data points minus one.
If *domain* has the same type and size as *data* and contains custom values for the x-axis, we can also use.
```cpp
	plotter.plot(domain, data);
```

<b>2D data</b><br />
If *data* is an *std::vector<std::vector<double>>* or a two-dimensional @link Arrays Array@endlink, it can be plotted using
```cpp
	plotter.plot(data);
```

# External plotting scripts {#PlottingScripts}
Some @link Properties@endlink extracted on the None or Ranges format, and saved using the @link ImportaingAndExportingData FileWriter@endlink, can be plotted using predefined python scripts.
After installing TBTK, these plotting scripts are available immediately from the terminal using the syntax
```bash
	TBTKPlotProperty.py File.h5 parameters
```
Here *File.h5* is the file to which the FileWriter has written the Property.

These scripts are immediately useful in a few different cases, but are more generally useful as templates.
To customize these templates, begin by copying the corresponding file from TBTK/Visualization/python/ into the applications source directory.

## Examples

<b>Density</b><br />
If the @link TBTK::Property::Density Density@endlink has been extracted on a two-dimensional grid, it can be plotted using
```cpp
	TBTKPlotDensity.py File.h5
```

<b>DOS</b>
```cpp
	TBTKPlotDOS.py File.h5 sigma
```
Here *sigma* is a decimal number specifying the amount of Gaussian smoothing to apply along the energy axis.

<b>EigenValues</b>
```cpp
	TBTKPlotEigenValues.py File.h5
```
This will plot the eigenvalues ordered from lowest to highest along the x-axis.

<b>LDOS</b><br />
If the @link TBTK::Property::LDOS LDOS@endlink has been extracted along a one-dimensional line, it can be plotted using
```cpp
	TBTKPlotLDOS.py File.h5 sigma
```
Here *sigma* is a decimal number specifying the amount of Gaussian smoothing to apply along the energy axis.

<b>Magnetization</b><br />
If the @link TBTK::Property::Magnetization Magnetization@endlink has been extracted on a two-dimensional grid, it can be plotted using
```cpp
	TBTKPlotMagnetization.py File.h5 theta phi
```
Here *theta* and *phi* are the polar and azimuthal angles, respectively, of the spin-polarization axis of interest.

<b>SpinPolarizedLDOS</b><br />
If the @link TBTK::Property::SpinPolarizedLDOS SpinPolarizedLDOS@endlink has been extracted along a one-dimensional line, it can be plotted using
```cpp
	TBTKPlotSpinPolarizedLDOS.py File.h5 theta phi sigma
```
Here *theta* and *phi* are the polar and azimuthal angles, respectively, of the spin-polarization axis of interest.
Further, *sigma* is a decimal number specifying the amount of Gaussian smoothing to apply along the energy axis.
