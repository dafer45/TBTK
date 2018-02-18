Manual {#manual}
======

- @subpage Introduction
- @subpage Overview
- @subpage UnitHandler
- @subpage Indices
- @subpage Model
- @subpage Solvers
- @subpage PropertyExtractors

@page Introduction Introduction

# Origin and scope {#OriginAndScope}
TBTK (Tight-Binding ToolKit) originated as a toolkit for solving tight-binding models.
However, the scope of the code has expanded beyond the area implied by its name, and is today best described as a library for building applications that solves second-quantized Hamiltonians with discrete indices  

<center>\f$H = \sum_{\mathbf{i}\mathbf{j}}a_{\mathbf{i}\mathbf{j}}c_{\mathbf{i}}^{\dagger}c_{\mathbf{j}} + \sum_{\mathbf{i}\mathbf{j}\mathbf{k}\mathbf{l}}V_{\mathbf{i}\mathbf{j}\mathbf{k}\mathbf{l}}c_{\mathbf{i}}^{\dagger}c_{\mathbf{j}}^{\dagger}c_{\mathbf{k}}c_{\mathbf{l}}\f$.</center>

In fact, even more general interaction terms than the one displayed above is possible to model.
However, TBTK does not yet have extensive support for solving interacting models and this manual is therefore focused on the non-interacting case.

Most quantum mechanical systems are described by Hamiltonians which at least involve a differential equation in some continuous variable or second quantized operators with a mix of discrete and continuous indices.
The restriction to discrete indices may therefore seem to severely limit the applicability of the library to a few special types of problems.
However, we note that it is very common for quantum mechanical problems with continuous variables to be described by discrete indices after proper analytical manipulations of the expressions.
For example, the continuous Schrödinger equation in a central potential with three continuous spatial coordinates does after some analytic manipulations become a problem with two discrete indices corresponding to the spherical harmonics, and one remaining continuous variable.
Moreover, if not handled analytically, any remaining continuous variable eventually has to be discretized.

Given that the restriction to discrete indices does not imply any severe limitation to the types of problems that can be addressed, a remaining challange is to allow for problems with arbitrary index complexity to be modeled and solved.
TBTK solves this through a flexible and efficient index system, which combined with a sophisticated storage structure for the Hamiltonian allows for essentially arbitrary index structures to be handled without significant performance penalties compared to highly optimized single purpose code.
In fact, TBTK is explicitly designed to allow solution algorithms to internally use whatever data structures that are best suited for the problem at hand to optimize the calculation, without affecting the format on which the model is specified or properties are extracted.

# Algorithms and data structures {#AlgorithmsAndDataStructures}

When writing software it is natural to think in terms of algorithms.
This is particularly true in scientific computation, where the objective most of the time is to carry out a set of well defined operations on an input state to arrive at a final answer.
Algorithm centered thinking is manifested in the imperative programing paradigm and is probably the best way to learn the basics of programming and to implement simple tasks.
However, while algorithms are of great importance, much of the success of todays computer software can be attributed to the development of powerful data structures.

Anyone who has writen software that is more than a few hundred lines of code knows that a major challange is to organize the code in such a way that the complexity do not scale up with the size of the project.
Otherwise, when e.g. coming back to a project after a few months, it may be difficult to make modifications to the code since you do not remember if or how it will affects other parts of the code.
The reason for this can largly be traced back to the lack of proper attention paid to data structures.
In particular, well designed data structures enables abstraction and encapsulation and is a core component of the object oriented programming paradigm.

Abstraction is the process of dividing code into logical units that aids the thinking process by allowing the programmer to think on a higher level.
Effective abstraction allows the programmer to forget about low level details and focus on the overarching problem at hands.
For some analogies, mathemtics and physics are rife with abstractions: derivatives are defined through limits but differential equations are written using derivative symbols, matrices are rectangles of numbers but we carry out much of the algebra manipulating letters representing whole matrices rather than individual matrix elements, etc.
Much of mathematics is ultimately just addition, subtraction, multiplication, and division of real numbers, but through abstraction problems of much greater complexity can be tackled than if everything was formulated with those four symbols alone.
Similarly programming is nothing but loops, conditional execution, assignment of values, function calls etc., but further abstraction allows the programmers mind to be freed from low level details to focus on the higher level aspects of much more complex problems.

While abstraction means dividing code into logical units that allows the programmer to think on a higher level, encapsulation means making those units largly independent of each other.
Different parts of a program should of course interact with each other, but low level details of a specific component should to an as large degree as posible be invisible to other components.
Instead of allowing (and requiring) other components of a code to manipulate its low level details, components should strive to present themself to other components through an easy to use interface.
The interface is provided through a so called application programming interface (API).
The API is essentially a contract between a component and the outside world, where the component specifies a promisse to solve a particular problem given a particular input.
Encapsulation makes it possible to update a piece of code without remembering what other parts of the code is doing, as long as the update respects the contract specified in the API, and is key to solving the scalability issue.
Developers mainly experienced with imperative programming likely recognize some of these concepts as being emboddied in the idea of dividing code into functions.
Object oriented programming greatly extends this powerful technique.

Scientific computing is often computationally intensive and much thought therefore goes into the development of different algorithms for solving the same problem.
Different algorithms may have their own strengths and weaknesses making them the preffered choice under different circumstances.
Often such algorithms are implemented in completely different software packages with little reuse of code, even though the code for the main algorithm may be a small part of the actual code.
This is a likely result when data structures are an afterthought and results in both replicated work and less reliable code since code that is reused in multiple projects is much more extensively tested.
A key to handling situations like this is called polymorphis and is a principle whereby different components essentially provides identical or partly identical contracts to the outside world, even though they internally may work very differently.
This allows for components to be used interchangeably with little changes to the rest of the code base.

TBTK is first and foremost a collection of data structures intended to enable the implementation of algorithms for solving quantum mechanical problems, but also implements several different algorithms for solving specific problems.

# c++11: Performance vs. ease of use {#cpp11PerformanceVsEaseOfUse}

Scientific computations are often very demanding and high performance is therefore often a high priority.
However, while low level programming languages offer high performance, they also have a reputation of being relatively difficult to work with.
A comparatively good understanding of the low level details of how a computer works is usually required to write a program in languages such as c/c++ and FORTRAN compared to e.g. MATLAB and python.
However, while c++ provides the ability to work on a very low level, it also provides the tools necessary to abstract away much of these details.
A well writen library can alleviate many of these issues, such as for example putting little requirement on the user to manage memory (the main source of errors for many new c/c++ programmers).
Great progress in this direction was taken with the establishment of the c++11 standard.
The language of choice for TBTK has therefore been c++11, and much effort has gone into developing data structures that is as simple as possible to use.
Great care has also been taken to avoid having the program crash without giving error messages that provide information that helps the user to resolve the problem.

@page Overview Overview

# Model, Solvers, and PropertyExtractors {#ModelSolversAndPropertyExtractors}
It is useful to think of a typical scientific numerical study as involving three relatively separate decissions:
- What is the model?
- What method to use?
- What properties to calculate?

When writing code, the answer to these three questions essentially determines the input, algorithm, and output, respectively.
To succesfully cary out studies of complex problems, it is important that it is easy to set up the model and to extract the properties, and that the underlying algorithm that performs the work is efficient.
However, the simultaneous requirement on the algorithm to be efficient and that the calculation is easy to set up easily run countrary to each other.
Efficiency often require low level optimization in the algorithm, which e.g. can put strict requirment of how the input and output is represented in memory.
If this means the user is required to setup the input and extract the output on a format that requires deep knowledge about the internal workings of the algorithm, two important problems arise.
First, if details about the algorithm is required to be kept in mind at all levels of the code it hinders the user from thinking about the problem on a higher level where numeric nuisance has been abstracted  away.
Second, if the specific requirements of an algorithm determines the structure of the whole program, the whole code has to be rewritten if the choice is made to try another algorithm.

To get around these problems TBTK is designed to encourage a workflow where the three stages of specifying input, chosing algorithm, and extracting properties are largly independent from each other.
To achieve this TBTK has a class called a Model that allows for general models to be setup.
Further, algorithms are implemented in a set of different Solvers, which takes a Model and internally converts it to the format most suitable for the algorithm.
Finally, the Solver is wrapped in a PropertyExtractor, where the different PropertyExtractors have a uniform interface.
By using the PropertyExtractors to extract properties from the Model, rather than by calling the Solvers directly, most code do not need to be changed if the Solver is changed.

# Auxiliary tasks
While the three questions above captures the essence of a typical scientific problem, auxiliary tasks such as reading/writing data from/to file, plotting, etc. are required to solve a problem.
TBTK therefore also have a large set of tools for simplifying such tasks, allowing the developer to put more mental effort into the main scientific questions.
Further, a fundamental key feature of TBTK is that it also comes with a powerful method for handling units.
While physical quantities often are expressed in terms of some specific combinations of units such as K, eV, C, etc. it is often useful to work in some for the problem at hands natural set of units.
In high-energy physics this may mean \f$ \hbar = c = 1\f$, while in condensed matter physics it can be useful to express energies in terms of Rydbergs or some arbitrary unit set by a hopping parameter.
For this TBTK provides a UnitHandler that enables the developer to specify the natural units for the given problem.
All function calls to TBTK functions should be understood to be in terms of the specified natural units.

# Implementing applications {#ImplementingApplications}
For developers interested in implementing calculations that are meant to answer specific physical questions, which is also refered to as implementing applications, TBTK comes ready with a set of native Solvers.
This manual is mainly intended to describe this use case and therefore covers the most important classes needed to achieve this.
In particular, this manual outlines how to properly setup a Model, select a Solver, and to extract properties using PropertyExtractors.

# Implementing new Solvers {#ImplementingNewSolvers}
TBTK is also intended to enable the development of new Solvers and provides many classes ment to simplify this task.
This manual does not cover all these classes and the interested developer is instead refered to the API for a more detailed description.
However, developers are still encouraged to study this manual to understand the design philosophy behind TBTK, and to also use the already existing Solvers as inspiration when writing new Solvers.
Doing so can significantly reduce the amount of overhead required to create a new Solver and makes it easier for other developers to use the new Solver in their own project.
The development of new Solvers are greatly encouraged and if you are interested in doing so but have any questions, please contact Kristofer Björnson at kristofer.bjornson@physics.uu.se.
New Solvers can either be released as stand alone packages or be pulled into the TBTK library.
In the later case the Solver will have to adhere to the main development philosophy of TBTK, but we are happy with providing help with polishing the Solver to the point that it does.

@page UnitHandler UnitHandler

# Units and constants {#UnitsAndConstants}
Most quantities of interest in physics have units, which means that the numerical value of the quantity depends on which units it is measured in.
Different sets of units are relevant in different situations, e.g. is meter (m) a relevant unit for length in macroscopic problems, while Ångström (Å) is more relevant on atomic scales.
However, computers work with unitless numbers, which means that any piece of code that relies on hard coded numerical values for physical constants implicitly force the user to work in the same set of units.
This is unacceptable for a library such as TBTK which aims at allowing physicist with different preferences for units to use the library to implement their own calculations.
It is also very useful if the library can supply such constants in whatever units the developer prefer.
To solve these issues, TBTK provides a UnitHandler that allows the user to specify what units are natural to the problem at hands, and all numbers passed to TBTK functions are assumed to be given in these natural units.

# Base units {#BaseUnits}
The UnitHandler borrows its terminology from the SI standard for units.
Not by forcing the user to work in SI units, but rather through a clear division of units into base units and derived units.
To understand what this means, consider distances and times.
These are quantities that are defined independently from each other and can be measured in for example meters (m) and seconds (s).
In comparison, a velocity is a measure of distance per time and cannot be defined independently of these two quantities.
Velocity can therefore be considered to be a quantity that is derived from the more fundamental quantities distance and time.
In principle, there is no reason why a given quantity has to be considered more fundamental than any other, and it is perfectly valid to e.g. view time as a quantity derived from the more fundamental quantities distance and velocity.
However, the fact remains that for the three quantities distance, time, and velocity, only two at a time can be defined independently from each other.
Further, among all the quantities encountered in physics, only seven can be defined independently from each other.
By fixing seven such quantities, a set of seven corresponding base units can be defined.
All other units are considered derived units.

The UnitHandler defines the fundamental quantities to be temperature, time, length, energy, charge, and count (amount).
This is the first point where the UnitHandler deviates from the SI system since the SI system do not define the units for energy and charge as base units, but instead the units for mass, current, and luminosity.
Note in particular that the UnitHandler currently only defines base units for six different quantities.
The missing quantity is due to an ambiguity regarding whether an angle should be considered a unitfull or unitless quantity.
Units for angle may therefore be added to the UnitHandler in the future.
The decission to make the units for energy and charge base units, rather than mass and current as in the SI system, is based on a subjective perception of the former being more generally relevant in quantum mechanical calculations.

Next, the UnitHandler also deviates from the SI system by only fixing the base quantities rather than the base units.
While e.g. the SI unit for length is meter (m), the UnitHandler allows the base unit for length to be set to a range of different units such as meter (m), millimiter (mm), nanometer (nm), Ångström (Å), etc.
Similarly a range of different options are available for other quantities, such as for example Joule (J) and electronvolt (eV) for energy, and Coulomb (C) and elementary charge (e) for charge.

By default the base units are
| Quantity    | Default base unit  | UnitHandler symbol |
|-------------|--------------------|--------------------|
| Temperature | K (Kelvin)         | Temperature        |
| Time        | s (seconds)        | Time               |
| Length      | m (meter)          | Length             |
| Energy      | eV (electron Volt) | Energy             |
| Charge      | C (Coulomb)        | Charge             |
| Count       | pcs (pieces)       | Count              |

Further, the available base units are
| Quantity    | Available base units                             |
|-------------|--------------------------------------------------|
| Temperature | kK, K, mK, uK, nK                                | 
| Time        | s, ms, us, ns, ps, fs, as                        |
| Length      | m, mm, um, nm, pm, fm, am, Ao                    |
| Energy      | GeV, MeV, keV, eV, meV, ueV, J                   |
| Charge      | kC, C, mC, uC, nC, pC, fC, aC, Te, Ge, Me, ke, e |
| Count       | pcs, mol                                         |

Most of these units should be self-explanatory, with Gx, Mx, kx, mx, etc. corresponds to giga, mega, kilo, milli, etc.
Further, Ao corresponds to Ångström (Å), while pcs corresponds to pieces.
If further base units are wanted, please do not hesitate to request additions.

If base units other than the default base units are wanted, it is recommended to set these at the very start of a program.
For example at the first line in the main routine.
This avoids ambiguities that results from changing base units in the middle of execution.
To for example set the base units to mK, ps, Å, meV, C, and mol, type
```cpp
	UnitHandler::setTemperatureUnit(UnitHandler::TemeratureUnit::mK);
	UnitHandler::setTimeUnit(UnitHandler::TimeUnit::ps);
	UnitHandler::setLengthUnit(UnitHandler::LengthUnit::Ao);
	UnitHandler::setEnergyUnit(UnitHandler::EnergyUnit::meV);
	UnitHandler::setChargeUnit(UnitHandler::ChargeUnit::C);
	UnitHandler::setCountUnit(UnitHandler::CountUnit::mol);
```

# Natural units {#NaturalUnits}
It is common in physics to use natural units in which for example \f$\hbar = c = 1\f$.
Such natural units simplify equations and allows mental effort to be concentrated on the physical phenomena rather than numerical details.
The same is true when implementing numerical calculations and it is for example common in tight-binding calculations to measure energy in units of some hopping parameter \f$t = 1\f$, while the actual unitfull value can be some arbitrary value such as \f$t = 724meV\f$.
In TBTK all function calls are performed in natural units, except for the UnitHandler calls that specifies the natural units.
This means that if the natural energy unit is set to e.g. \f$724meV\f$, an energy variable with say the value 1.2 that is passed to a function is internally interpretted by TBTK to have the unitfull value \f$1.2\times724meV\f$.
However, note that this conversion is not necessarily done at the point where the function call is made and may be repeatedly done at later points of execution if the variable is stored internally.
This is why it is important to not reconfigure the UnitHandler in the middle of a program since this introduces ambiguities.

The natural unit is also known as the scale of the problem, and the code required to specify the natural energy unit (scale) to be \f$724meV\f$ is
```cpp
	//Set the energy base unit
	UnitHandler::setEnergyUnit(UnitHandler::EnergyUnit::meV);
	//Set the natural energy unit (scale) 
	UnitHandler::setEnergyScale(724);
```
The code for setting the other five natural units is similar, with the word 'Energy' exchanged for the relevant quantity.

# Converting between base and natural units {#ConvertingBetweenBaseAndNaturalUnits}
Because the input and output from TBTK functions are in natural units, it is convenient to have a simple way to convert between the two.
The UnitHandler provides such functions through a set of functions on the form
```cpp
	double quantityInBaseUnits    = UnitHandler::convertQuantityNtB(quantityInNaturalUnits);
	double quantityInNaturalUnits = UnitHandler::convertQuantityBtN(quantityInBaseUnits);
```
Here 'Quantity' is to be replace by the corresponding UnitHandler symbol specified in the table above, and NtB and BtN should be read 'natural to base' and 'base to natural', respectively.

# Derived units {#DerivedUnits}
Since derived units are defined in terms of the base units, it is in principle possible to use the above method to perform conversion of arbitrary derived units to and from natural units.
However, doing so would require decomposing the derived unit into the corresponding base units, convert the base units one by one, multiply them together with the appropriate exponents, and finally multiply the quantity itself by the result.
Moreover, even though it e.g. may be most convenient to work in the base units \f$eV\f$, \f$m\f$, and \f$s\f$ for energy, length, and time, in which case the corresponding mass unit is \f$eVs^2/m^2\f$, it may be more convenient to actuall specify mass using the unit \f$kg\f$.
For this reason the UnitHandler aslo has special support for certain derived units.
Currently this is restricted to mass and magnetic field strength, but if more units are wanted, please do not hesitate to request additional derived units.
The full list of possible derived units are
| Quantity                | Available derived units                      | UnitHandler symbol |
|-------------------------|----------------------------------------------|--------------------|
| Mass                    | kg, g, mg, ug, ng, pg, fg, ag, u             | Mass               |
| Magnetic field strength | MT, kT, T, mT, uT, nT, GG, MG, kG, G, mG, uG | MagneticField      |

To convert mass, say specified in the derived units \f$kg\f$ to and from base and natural units the following function calls can be made
```cpp
	double massInBaseUnits    = UnitHandler::convertMassDtB(massInDerivedUnits, UnitHandler::MassUnit::kg);
	double massInNaturalUnits = UnitHandler::convertMassDtN(massInDerivedUnits, UnitHandler::MassUnit::kg);
	double massInDerivedUnits = UnitHandler::convertMassBtD(massInBaseUnits,    UnitHandler::MassUnit::kg);
	double massInDerivedUnits = UnitHandler::convertMassNtD(massInNaturalUnits, UnitHandler::MassUnit::kg);
```
Here DtB, DtN, BtD, and NtD should be read 'derived to base', 'derived to natural', 'base to derived', and 'natural to derived', respectively.
The function calls mimic the six corresponding combinations of calls for conversion between base and natural units, with the exception that for derived units the actual derived unit has to be passed as a second argument.

# Constants {#Constants}
The specification of physical constants is prone to errors.
Partly because physical constants more often than not are long strings of rather random digits, and partly because it is easy to make mistakes when converting the constants to the particular units used in the calculation.
The UnitHandler alleviates this issue by providing a range of predefined constants that can be requested on the currently used base or natural units.
The available constants are
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
Please do not hesitate to request further constants.

Once the base and natural units have been specified using the calls described above, the physical constants can be requested using function calls on the form
```cpp
	double constantValueInBaseUnits    = UnitHandler::getSymbolB();
	double constantValueInNaturalUnits = UnitHandler::getSymbolN();
```
Here 'Symbol' is to be replaced by the corresponding symbol listed under 'UnitHandler symbol' in the table above.

# Unit strings {#UnitStrings}
When printing values, it is useful to also print the actuall unit strings.
The UnitHandler therefore also provides methods for requesting the base unit strings for different quantities, which is obtained through function calls on the form
```cpp
	string unitSring = UnitHandler::getSymbolUnitString();
```
Here 'Symbol' can be any of the symbols listed in the tables over base units, derived units, and constants.

@page Indices Indices

# Complex index structures {#ComplexIndexStructures}
To get an idea about the generality of problems that TBTK is intended to be able to handle, imagine a molecule on top of a graphene sheet, which in turs sits on top of a three-dimensional magnetic substrate material.
Further assume that we can model the three individual systems in the following way
- Molecule: A one-dimensional chain with three orbitals and two spins per site.
- Graphene: A two-dimensional sheet with two atoms per unit cell and with a single orbital and tow spins per atom.
- Substrate: A three-dimensional bulk with three atoms per unit cell, five orbitals per atom, and a single relevant spin-species.

To describe a system like this, we need three types of operators with three different types of index structure.
We can for example introduce the operators \f$a_{xo\sigma}\f$, \f$b_{xys\sigma}\f$, and \f$c_{xyzso}\f$ for the molecule, graphene, and substrate, respectively.
Here \f$x\f$, \f$y\f$, and \f$z\f$ correpsonds to spatial coordinates, \f$s\f$ is a sublattice index, \f$o\f$ is an orbital index, and \f$\sigma\f$ is a spin index.
First we note that the number of indices per operator is not the same.
Further, even though every operator have an \f$x\f$-index, there is not necesarily any actual relationship between the different \f$x\f$-indices.
In particular, the molecule may be oriented along some axis which does not coincide with the natural coordinate axes of the other two materials, and even more importantly, there is no reason why \f$x\f$ should run over the same number of values in the different materials.
Rather, the \f$x\f$-index is just a symbol indicating the first spatial index for the corresponding operator.
Similarly, the sublattice and orbital indices for the different operators are not the same.

In TBTK systems like this can easily be modeled using flexible indices that compounds a set of indices into a list of non-negative integers in curly braces.
Such a compund index is usually refered to simply as an index, while the individual components are refered to as subindices.
While we above used different letters to differentiate between operators with different index structures, in TBTK this is instead handled by introducing one more subsystem identifier at the front of the list of subindices.
E.g. can can we write typical indices as (spin up = 0, spin down = 1)
| Index              | Description                                                                |
|--------------------|----------------------------------------------------------------------------|
| {0, 1, 0, 1}       | Molecule (subsystem 0), site 1, orbital 0, spin down                       |
| {1, 2, 3, 0, 0}    | Graphene (subsystem 1), unit cell (2, 3), sublattice site 0, spin up       |
| {2, 3, 2, 1, 2, 3} | Substrate (subsystem 2), unit cell (3, 2, 1), sublattice site 2, orbital 3 |

# Limitations on the index structure {#LimitationsOnTheIndexStructure}
We have already mentioned one limitations on the indices, which is that they have to be non-negative numbers.
Although this is no real restriction on what types of problems that can be modeled, negative indices abounds in quantum mechanics and this is certainly not without inconvenience.
However, the optimizations and additional features that are enabled by this design decision has been deemed far more important in this case.
Nevertheless, support for negative indices could be added in the future through additional "syntactic suggar".
Any developer interested in pursuing this direction is most welcome to discuss these ideas.

Another restriction has to do with the fact that the subsystem index was added at the front of the indices.
This is not strictly required, at least in case the number of \f$x\f$-indices are the same for all three operators.
TBTK requires that systems only differ in their index structure to the right of a subindex for which they have different values.
That is, by letting the first subsystem index be 0, 1, and 2 for the three different systems, the rest of the index structure for the three indices can be completely different.

To be more precise what it means for two systems to have different subindex structure: two systems differ in their subindex structure if some specific subindex runs over different number of values for the two systems, or if a particular subindex exist in one system but not in the other.
This puts no real restriction on the types of problems that can be solved.
However, more tricky situations than the one above can arise.
Let us for example consider the case where the substrate above has different number of orbitals for the different sublatice sites.
In this case it is an error to write the indices on the form {subsystem, x, y, z, orbital, sublattice}, because the orbitals run over different numbers of orbitals even though they can have identical values for all the subindices to the left of the orbital subindex.
The original subindex order {subsystem, x, y, z, sublattice, orbital} given above does not have this problem though, since the sublattice index stands to the left of the orbital subindex and is different for the different sites.
In general, ordering the subindices with "less local" subindices to the left should almost allways resolve such issues.

@page Model Model

# The Model class {#TheModelClass}
The main class for setting up a model is the Model class, which acts as a container for model specific parameters such as the Hamiltonian, temperature, chemical potential, etc.
For a simple example, lets consider a simple two level system described by the Hamiltonian
<center>\f$H = U_{0}c_{0}^{\dagger}c_{0} + U_{1}c_{1}^{\dagger}c_{1} + V\left(c_{0}^{\dagger}c_{1} + H.c.\right)\f$,</center>
which is to be studied at T = 300, and zero chemical potential.
Before setting up the model we write the Hamiltonian on the canonical form
<center>\f$H = \sum_{\mathbf{i}\mathbf{j}}a_{\mathbf{i}\mathbf{j}}c_{\mathbf{i}}^{\dagger}c_{\mathbf{j}}\f$,</center>
where \f$a_{00} = U_0\f$, \f$a_{11} = U_1\f$, and \f$a_{01} = a_{10} = V\f$.
Once on this form, the model is ready to be fed into a Model object, which is achieved as follows
```cpp
	double U_0 = 1;
	double U_1 = 2;
	double V = 0.5;

	Model model;
	model.setTemperature(300);
	model.setChemicalPotential(0);
	model << HoppingAmplitude(U_0, {0}, {0});
	model << HoppingAmplitude(U_1, {1}, {1});
	model << HoppingAmplitude(V,   {0}, {1}) + HC;
	model.construct();
```
We first make a note about the terminology used in TBTK.
From the canonical form for the Hamiltonian, and if we write the time evolution of a state as
<center>\f$\Psi(t+dt) = (1 - iHdt)\Psi(t)\f$,</center>
it is clear that \f$a_{\mathbf{i}\mathbf{j}}\f$ is the amplitude associated with the process where an electron is removed from \f$\mathbf{j}\f$ and inserted at \f$\mathbf{i}\f$.
That is, in tight-binding terminology, the amplitude associated with a particle hopping from \f$\mathbf{j}\f$ to \f$\mathbf{i}\f$.
While the term hopping amplitude often is restricted to the case where the two indices actually differ from each other, in TBTK it is used to refer to any \f$a_{\mathbf{i}\mathbf{j}}\f$.
Moreover, the indices \f$\mathbf{i}\f$ and \f$\mathbf{j}\f$ are often refered to as to- and from-indices, respectively, and a HoppingAmplitude is created as
```cpp
	HoppingAmplitude(value, toIndex, fromIndex);
```
In the example above two such HoppingAmplitudes are created for the diagonal entries and added to the Model using the <<-operator.
Similarly, in the second last line a HoppingAmplitude that corresponds to \f$a_{01}\f$ is created and added to the Model.
However, on close inspection we see that + *HC* is added at the end of the line, which means that also the Hermitian conjugate is added.
The last line that reads
```cpp
	model.construct();
```
creates a mapping between the indices that have been added to the Model and a linear Hilbert space index.
It is important that the this call is performed after the HoppingAmplitudes has been added to the Model and that no more HoppingAmplitudes are added after this point.

# Physical indices and Hilbert space indices {#PhysicalIndicesAndHilbertSpaceIndices}
Although the above example is too simple to make the point since the index structure is trivial, we know from the Indices chapter that TBTK allows also much more complicated index structures to be used.
One of the core features of TBTK is embodied in the call to *model.construct()*.
To understand why, we note that although the indices in the problem above have a linear structure, this is rarely the case in general.
In other problems, such as for example a two-dimensional lattice the index structure is not linear but is easily linearized by a defining a mapping such as \f$h = L_y x + y\f$.
The indices \f$(x, y)\f$ or {x, y} in TBTK notation we will call physical indices, while linearized indices such as \f$h\f$ are refered to as Hilbert space indices.

So why do we need linearization?
One answer is that algorithms almost universally are going to be most efficient in a linear basis.
Moreover, algorithms can also be made much more general if the linearization procedure is not part of the algorithm itself.
If mappings such as \f$h = L_y x + y\f$ are hard coded into the Solver, it is essentially a single purpose code, possibly with a few nobs to turn to adjust the system size and parameter values, but the code is locked to a very specific type of problem.
On the contrary, a Solver that accepts any Model that can provide itself on a linearized form can be reused for widely different types of problems.

So why physical indices?
Is it not better if the developer performs the linearization from the start to improve performance?
These questions would have been relevant if performance was an issue.
The mapping from physical indices to Hilbert space and back again is certainly not a computationally cheap task if implemented through a naive approach such as a lookup table.
In fact, this would have been prohibitively expensive.
One of the most (or simply the most) important core data structures of TBTK is a tree structure in which the HoppingAmplitudes are stored.
It allows for conversion back and forth between physical indices and Hilbert space indices with an overhead cost that is negligible in most parts of the code.
Taking the responsibilty to linearize indices of from the developer is a significant abstraction that allows mental capacity to be focused on physics instead of numerics.

The one place where the overhead really is important is in the Solvers where most of the computational time usually is spent.
Solvers therefore usually internally convert the Model to some much more optimal format using the linearization provided by the Model before starting the main calculation.
This is also the reason why, as described in the PropertyExtractor chapter, it is recommended to not use the Solvers immediately to extract properties, but to instead wrapp them in PropertyExtractors.
The PropertyExtractors provides the interface through which properties can be extracted from the Solvers using physical indices.
In short, the distinction between physical indices and Hilbert space indices allows application developers to focus on the physics of the particular problem, while simultaneously allowing Solver developers to focus on numerical details and general properties rather than system specific details.

# Example: A slightly more complex Model {#ExampleASlightlyMoreComplexModel}
To appreciate the ease with which Models can be setup in TBTK, we now consider a slightly more complex example.
Consider a magnetic impurity on top of a two-dimensional substrate of size 51\times 51 described by
<center>\f$H = H_{s} + H_{Imp} + H_{Int}\f$</center>
where
<center>\f{eqnarray*}{
	H_{s} &=& U_s\sum_{\mathbf{i}\sigma}c_{\mathbf{i}\sigma}^{\dagger}c_{\mathbf{i}\sigma} - t\sum_{\langle\mathbf{i}\mathbf{j}\rangle}c_{\mathbf{i}\sigma}^{\dagger}c_{\mathbf{j}\sigma},\\
	H_{Imp} &=& (U_{Imp} - J)d_{\uparrow}^{\dagger}d_{\uparrow} + (U_{Imp} + J)d_{\downarrow}^{\dagger}d_{\downarrow},\\
	H_{Int} &=& \delta\sum_{\sigma}c_{(25,25)\sigma}^{\dagger}d_{\sigma} + H.c.
\f}</center>
Here \f$\mathbf{i}\f$ is a two-dimensional index, \f$\langle\mathbf{i}\mathbf{j}\rangle\f$ indicates summation over nearest neighbors, \f$\sigma\f$ is a spin index, and \f$c_{\mathbf{i}}\f$ and \f$d_{\sigma}\f$ are operators on the substrate and imputiry, respectively.
The parameters \f$U_s\f$ and \f$U_{Imp}\f$ are onsite energies on the substrate and impurity, respectively, while \f$t\f$ is a nearest neighbor hopping amplitude, \f$J\f$ is a Zeeman term, and \f$\delta\f$ is the coupling strength between the substrate and impurity.

We first note that an appropriate index structure is {0, x, y, s} for the substrate and {1, s} for the impurity.
Using this index structure we next tabulate the hopping parameters on the canonical form given at the beginning of this chapter.
| Hopping amplitude                          | Value         | To Index         | From Index       |
|--------------------------------------------|---------------|------------------|------------------|
| \f$a_{(0,x,y,\sigma),(0,x,y,\sigma)}\f$    | \f$U_s\f$     | {0,   x,   y, s} | {0,   x,   y, s} |
| \f$a_{(0,x+1,y,\sigma),(0,x,y,\sigma)}\f$  | \f$-t\f$      | {0, x+1,   y, s} | {0,   x,   y, s} |
| \f$a_{(0,x,y,\sigma),(0,x+1,y,\sigma)}\f$  | \f$-t\f$      | {0,   x,   y, s} | {0, x+1,   y, s} |
| \f$a_{(0,x,y+1,\sigma),(0,x,y,\sigma)}\f$  | \f$-t\f$      | {0,   x, y+1, s} | {0,   x,   y, s} |
| \f$a_{(0,x,y,\sigma),(0,x,y+1,\sigma)}\f$  | \f$-t\f$      | {0,   x,   y, s} | {0,   x, y+1, s} |
| \f$a_{(1,\sigma),(1,\sigma)}\f$            | \f$U_{Imp}\f$ | {1, s}           | {1, s}           |
| \f$a_{(1,\uparrow),(1,\uparrow)}\f$        | \f$-J\f$      | {1, 0}           | {1, 0}           |
| \f$a_{(1,\downarrow),(1,\downarrow)}\f$    | \f$J\f$       | {1, 1}           | {1, 1}           |
| \f$a_{(0,25,25,\sigma),(1,\sigma)}\f$      | \f$\delta\f$  | {0,  25,  25, s} | {1, s}           |
| \f$a_{(1,\sigma),(0,25,25,\sigma)}\f$      | \f$\delta\f$  | {1, s}           | {0,  25,  25, s} |

Symbolic subindices should be understood to imply that the values are valid for all possible values of the corresponding subindices.
We also note that hopping amplitudes that appear multiple times should be understood to add the final value.
For example does \f$a_{(1,\uparrow),(1,\uparrow)}\f$ appear twice (sixth and seventh row) and should be understood to add to \f$U - J\f$.
While the first column is the analytical representation of the symbol for the hopping amplitudes, the third and fourth column is the corresponding numerical represenation.
In particular, we note that we use 0 to mean up spin and 1 to mean down spin.
Next we note that the table can be reduced if we take into account that row 2 and 3, 4 and 5, and 9 and 10 are each others Hermitian conjugates.
Furtherher, row 7 and 8 can be combined into a single row by writing the value as \f$-J(1 - 2s)\f$.
The table can therefore if we also ignore the first column be compressed to
| Value            | To Index         | From Index       | Add Hermitian conjugate |
|------------------|------------------|------------------|-------------------------|
| \f$U_s\f$        | {0,   x,   y, s} | {0,   x,   y, s} |                         |
| \f$-t\f$         | {0, x+1,   y, s} | {0,   x,   y, s} | Yes                     |
| \f$-t\f$         | {0,   x, y+1, s} | {0,   x,   y, s} | Yes                     |
| \f$U_{Imp}\f$    | {1, s}           | {1, s}           |                         |
| \f$-J(1 - 2s)\f$ | {1, 0}           | {1, 0}           |                         |
| \f$\delta\f$     | {0,  25,  25, s} | {1, s}           | Yes                     |

Once on this form, it is simple to implement the Model as follows
```cpp
	const int SIZE_X = 51;
	const int SIZE_Y = 51;

	double U_s = 1;
	double U_imp = 1;
	double t = 1;
	double J = 1;
	double delta = 1;

	Model model;

	//Setup substrate.
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				model << HoppingAmplitude(U_s, {0, x, y, s}, {0, x, y, s});

				if(x+1 < SIZE_X)
					model << HoppingAmplitude(-t, {0, x+1, y, s}, {0, x, y, s}) + HC;
				if(y+1 < SIZE_Y)
					model << HoppingAmplitude(-t, {0, x, y+1, s}, {0, x, y, s}) + HC;
			}
		}
	}

	for(int s = 0; s < 2; s++){
		//Setup impurity.
		model << HoppingAmplitude(U_imp,      {1, s}, {1, s});
		model << HoppingAmplitude(-J*(1-2*s), {1, s}, {1, s});

		//Add coupling between the substrate and impurity.
		model << HoppingAmplitude(delta, {0, SIZE_X/2, SIZE_Y/2, s}, {1, s}) + HC;
	}

	//Construct model.
	model.construct();
```

For pedagogical reasons we here went through the process of converting the analytical Hamiltonian to a numerical Model in quite some detail.
However, with a small amount of experience the second table can be read of immediately from the Hamiltonian, cutting down the work significantly.
A key advice is to utilize the Hermitian conjugate to their maximum like we did above.
Note in particular that we used this to only have \f$x+1\f$ and \f$y+1\f$ in one position for the indices, respectivel (and no \f$x-1\f$ or \f$y-1\f$).
Doing so reduces the number of lines of code, improves readability, simplifies maintainance, and consequently reduces the chance of introducing errors.

# Advanced example: Using IndexFilters to construct a Model {#AdvancedExampleUsingIndexFiltersToConstructAModel}
While the already introduced concepts significantly simplifies the modeling of complex geometries, TBTK provides further ways to simplify the modeling stage.
In particular, we note that in the example above, conditional statements had to be used in the first for-loop to ensure that HoppingAmplitudes were not added accross the boundary of the system.
For more complex structures it is useful to be able to separate the specification of such exceptions to the rule from the specification of the rule itself.
This can be accomplished through the use of an IndexFilter, which can be used by the Model to accept or reject HoppingAmplitudes based on their to- and from-indices.

In this example we will show how to setup a simple two-dimensional sheet like the substrate above and in addition add a hole in the center of the substrate.
We will here assume that the size of the substrate and the radius of the hole has been specified using three global variables SIZE_X, SIZE_Y, and RADIUS.
A good implementation would certainly remove the need for global variables, but we use this method here to highlight the core concepts since removing the global variables requires a bit of extra code that obscures the key point.
Moreover, for many applications such parameters would be so fundamental to the calculation that it actually may be justified to use global variables to store them.

The first step toward using an IndexFilter is to create one.
The syntax for doing so is as follows.
```cpp
class Filter : public AbstractIndexFilter{
public:
	Filter* clone() const{
		return new Filter();
	}

	bool isIncluded(const Index &index){
		double radius = sqrt(
			pow(index[0] - SIZE_X/2, 2)
			+ pow(index[1] - SIZE_Y/2, 2)
		);
		if(radius < RADIUS)
			return false;
		else if(index[0] < 0 || index[0] >= SIZE_X)
			return false;
		else if(index[1] < 0 || index[1] >= SIZE_Y)
			return false;
		else
			return true;
	}
};
```
The experienced c++ programmer recognizes this as an implementation of an abstract base class and is encouraged to write more complicated filters.
In this case we note that it is important that the function *clone()* returns a proper copy of the Filter, which in this case is trivial since there are no member variables.
However, for the developer not familiar with such concepts, it is enough to view this as a template where the main action takes place in the function *isIncluded()*.
This function is responsible for returning whether a given input index is included in the Model or not.
As seen we first calculate the *radius* for the Index away from the center of the sheet.
Next we check whether the calculated *radius* is inside the specified *RADIUS* for the hole, or if it falls outside the boundaries of the sample.
If either of these are true we return false, while otherwise we return true.

<b>Note:</b> When writing a Filter it is important to take into account not only the Index structure of Indices that are going to be rejected, but all Indices that are passed to the Model at the point of setup.
If there for example would have been some Indices in the Model that only had one subindex, the Filter above would have been invalid since it may be passed such an Index and try to access *index[1]*.
When writing filters it is therefore important to ensure they are compatible with the Model as a whole and are able to respond true or false for every possible Index in the Model.
Since in this case we model a two-dimensional sheet where all Indices have more than two subindices this Filter is alright.

Once the Filter is specified, we are ready to use it to setup a Model
```cpp
	double U_s = 1;
	double t = 1;

	Model model;
	Filter filter;
	model.setFilter(filter);
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int s = 0; s < 2; s++){
				model << HoppingAmplitude(U_s, {x, y, s}, {x, y, s});
				model << HoppingAmplitude(-t, {x+1, y, s}, {x, y, s}) + HC;
				model << HoppingAmplitude(-t, {x, y+1, s}, {x, y, s}) + HC;
			}
		}
	}
	model.construct();
```

# Advanced: Callback functions
Sometimes it is useful to be able to delay specification of a HoppingAmplitudes value to a later time than the creation of the Model.
This is for example the case if the same Model is going to be solved multiple times for different parameter values, or if some of the parameters in the Hamiltonian are going to be determined self-consistently.
For this reason it is possible to pass a function as value argument to the HoppingAmplitude rather than a fixed value.
If we have indices with the structure {x, y, s}, where the last index is a spin, and there exists som global parameter \f$J\f$ that determines the current strength of the Zeeman term, a typical callbackFunction looks like
```cpp
	complex<double> callbackJ(const Index &to, const Index &from){
		//Get spin index.
		int s = from[2];

		//Return the value of the HoppingAmplitude
		return -J*(1 - 2*s);
	}
```
Just like when writing an IndexFilter, a certain amount of Model specific information needs to go into the specification of the callbacks.
Here we have for example chosen to determine the spin by looking at the 'from'-Index, which should not differ from the spin-index of the *to*-Index.
However, unlike when writing IndexFilters, the HoppingAmplitude callbacks only need to make sure that they work for the restricted Indices for which the callback is fed into the Model together with, since these are the only Indices that will ever be passed to the callback.

Once the callback is defined, it is possible to use it when setting up a model as follows
```cpp
	model << HoppingAmplitude(callbackJ, {x, y, s}, {x, y, s});
```

# Block structure {#BlockStructure}
One of the most powerful methods for solving quantum mechanical problems is block diagonalization of the Hamiltonian.
What this means is that the Hamiltonian is broken apart into smaller blocks that are decoupled from each other and thereby effectively replaces one big problem with many smaller problems.
The smaller problems are much simpler to solve and usually far outweighs the fact that several problems have to be solved instead.
The most obvious example of this is a Hamiltonian which when written in reciprocal space separates into independent problems for each momentum \f$\mathbf{k}\f$.

To facilitate the exploitation of block diagonal Hamiltonians TBTK has restricted support for automatically detecting when it is handed a Hamiltonian with a block diagonal structure.
However, for this to work the developer has to organize the Index structure in such a way that TBTK is able to automatically take advantage of the block diagonal structure.
Namely, TBTK will automaitcally detect existing block structures as long as the Index has the subindices that identifies the independen blocks standing to the left of the intra block indices.
For a system with say three momentum subindices, one orbital subindex, and one spin subindex, where the Hamiltonian is block diagonal in the momentum space subindices, an appropriate Index structure is {kx, ky, kz, orbital, spin}.
If for some reason the Index structure instead is given as {kx, ky, orbital, kz, spin}, TBTK will only be able to automatically detect kx and ky as block-indices, and will treate all of the three remaining subindices orbital, kz, and spin as internal indices for the blocks (at least as long as the Hamiltonian does not happen to also be block diagonal in the orbital subindex).

@page Solvers Solvers

# Limiting algorithm specific details from spreading {#LimitingAlgorithmSpecificDetailsFromSpreading}
Depending on the Model and the properties that are sought for, different solution methods are best suited to carry out different tasks.
Sometimes it is not clear what the best method for the given task is and it is useful to be able to try different approaches for the same problem.
However, different algorithms can vary widely in their approach to solving a problem, and specifically vary widely in how they need the data to be represented in memory to carry out their tasks.
Without somehow limiting these differences to a restricted part of the code the specific numerical details of a Solver easily spreads throughout the code and makes it impossible to switch one Solver for another without rewriting the whole code.

In TBTK the approach to solving this problem is to provide a clear separation between the algorithm implemented inside the Solvers, and the specification of the Model and the extraction of properties.
Solvers are therefore required to all accept a universal Model object and to internally convert it to whatever representation that is best suited to the algorithm.
Solvers are then wrappen in PropertyExtractors which further limits the Solver specific details from spreading to other parts of the code.
The idea is to limit the application developers exposure to the Solver as much as possible, freeing mental capacity to focus on the physical problem at hands.
Nevertheless, a certain amount of method specific configurations are inevitable and the appropriate place to make such manipulations is through the Solver interface itself.
This approach both limits the developers exposure to unecessary details, while also making sure the developer understands when algorithm specific details are configured.

Contrary to what it may sound like, limiting the developers exposure to the Solver does not mean conceiling what is going on inside the Solver and to make it an impenetrable black box.
In fact, TBTK aims at making such details as transparent as possible and to invite the interested developer to dig as deep as prefered.
To limit the exposure as much as possible rather means that once the developer has chosen Solver and configured it, the Solver specific details should not spread further and the developers should be free to not worry about low level details.
In order to chose the right Solver for a given task and to configure it efficiently it is useful to have an as good understanding as possible about what the algorithm actually is doing.
Therefore we here describe what the Solvers does, what their strengths and weaknesses are, and how to set up and configure them.
In the code examples presented here it is assumed that a Model object has already been created.

# Overview of native Solvers {#OverviewOfNativeSolvers}
TBTK currently contains four production ready Solvers.
These are called DiagonalizationSolver, BlockDiagonalizationSolver, ArnoldiSolver, and ChebyshevSolver.
The first two of these are based on diagonalization, allowing for all eigenvalues and eigenvectors to be calculated.
Their strength is that once a problem has been diagonalized, complete knowledge about the system is available and arbitrary properties can be calculated.
However, diagonalization scales poorly with system size and is therefore not feasible for very large systems.
The BlockDiagonalizationSolver provides important improvements in this regard for large systems if they are block diagonal, in which case the BlockDiagonalizationSolver can handle very large systems compared to the DiagonalizationSolver.

Next, the ArnoldiSolver is similar to the DiagonalizationSolvers in the sense that it calculates eigenvalues and eigenvectors.
However, it is what is know as an iterative Krylov space Solver, which succesively builds up a subspace of the Hilbert space and performs diagonalization on this restricted subspace.
Therefore the ArnoldiSolver only extracts a few eigenvalues and eigenvectors.
Complete information about a system can therefore usually not be obtained with the help of the ArnoldiSolver, but it can often give access to the relevant information for very large systems if only a few eigenvalues or eigenvectors are needed.
Arnoldi iteration is closely related to the Lanczos method and is also the underlying method used when extracting a limited number of eigenvalues and eigenvectors using MATLABS eigs-function.

Finally, the ChebyshevSolver is different from the other methods in that it extracts the Green's function rather than eigenvalues and eigenvectors.
The ChebyshevSolvers strenght is also that it can be used for relatively large systems.
Moreover, while the DiagonalizationSolvers requires that the whole system first is diagonalized, and thereby essentially solves the full problem, before any property can be extracted, the ChebyshevSolver allows for individual Green's functions to be calculated which contains only partial information about the system.
However, the ChebyshevMethod is also an iterative method (but not Krylov), and would in fact require an infinite number of steps to obtain an exact solution.

# General remarks about Solvers {#GeneralRemarksAboutSolvers}
The Solvers all innherit from a common base class called Solver.
This Solver class is an abstract class, meaning it is not actually possible to create an object of the Solver class itself.
However, the Solver base class forces every other Solver to implement a method for seting the Model that it should work on.
The following call is therefore possible (and required) to call to initialize any given Solver called *solver*
```cpp
	solver.setModel(model);
```
The Solver class also provides a corresponding *getModel()* function for retreiving the Model.
It is important here to note that although the Model is passed to the Solver and the Solver will remember the Model, it will not assume ownership of the Model.
It is therefore important that the Model remains in memory throughout the lifetime of the Solver and that the caller takes responsibility for any possible cleanup work.
The user not familiar with memory management should not worry though, as long as standard practices as outlined in this manual are observed, these issues are irrelevant.

The Solvers are also Communicatiors, which means that they may output information that is possible to mute.
This can become important if a Solver for example is created or executed inside of a loop.
In this case extensive output can be produced and it can be desired to turn this off.
To do so, call
```cpp
	solver.setVerbose(false);
```

# DiagonalizationSolver {#DiagonalizationSolver}
The DiagonalizationSolver sets up a dense matrix representing the Hamiltonian and then diagonalizes it to obtain eigenvalues and eigenvectors.
The DiagonalizationSolver is probably the simplest possible Solver to work with as long as the system sizes are small enough to make it feasible, which means Hilbert spaces with a basis size of up to a few thousands.
A simple set up of the DiagonalizationSolver
```cpp
	//Create a DiagoanlizationSolver.
	DiagonalizationSolver solver;
	//Set the Model to work on.
	solver.setModel(model);
	//Diagonalize the Hamiltonian
	solver.run();
```
That's it. The problem is solved and can be passed to a corresponding PropertyExtractor for further processing.

## Estimating time and space requirements
Since diagonalization is a rather simple problem conceptually, it is easy to estimate the memory and time costs for the DiagonalizationSolver.
Memorywise the Hamiltonian is stored as an upper triangular matrix with complex<double> entries each 16 bytes large.
The storage space required for the Hamiltonian is therefore roughly \f$16N^2/2 = 8N^2\f$ bytes, where \f$N\f$ is the basis size of the Hamiltonian.
Another \f$16N^2\f$ bytes are required to store the resulting eigenvectors, and \f$8N\f$ bytes for the eigenvalues.
Neglecting the storage for the eigenvalues the approximate memory footprint for the DiagonalizationSolver is \f$24N^2\f$ bytes.
This runs into the GB range around a basis size of 7000.

The time it takes to diagonalize a matrix cannot be estimated with the same precission since it depends on the exact specification of the computer that the calculations are done on, but as of 2018 runs into the hour range for basis sizes of a few thousands.
However, knowing the execution time for a certain size on a certain computer, the execution can be rather accurately predicted for other system sizes using that the computation time scales as \f$N^3\f$.

## Advanced: Self-consistency callback
Sometimes the value of one or several parameters that go into the Hamiltonian are not known a priori, but it is instead part of the problem to figure out the correct value.
A common approach for solving such problems is to make an initial guess for the parameters, solve the model for the corresponding parameters, and then update the parameters with the so obtained values.
If the problem is well behaved enough, such an approach results in the unknown parameters eventually converging to fixed values.
Once the calculated parameter value is equal (within some tollerance) to the input parameter in the current iteration, the parameters are said to have reached self-consistency.
That is, the calculated parameters are consistent with themself in the sense that if they are used as input parameters, they are also the result of the calculation.

When using diagonalization the self-consistent procedure is very straight forward: diagonalize the Hamiltonian, callculate and update parameters, and repeat until convergence.
The DiagonalizationSolver is therefore prepared to run such a self-consistency loop.
However, the the second step requires special knowledge about the system which the DiagonalizationSolver cannot incorporate without essentially becoming a single purpose solver.
The solution to this problem is to allow the application developer to supply a callback function that the DiagonalizationSolver can call once the Hamiltonian has been diagonalized.
This function is responsibile for calculating and updating relevant parameters, as well as informing the DiagonalizationSolver whether the solution has converged.
The interface for such a function is
```cpp
	bool selfConsistencyCallback(DiagonalizationSolver &solver){
		//Calculate and update parameters.
		//...

		//Determine wheter the solution has converged or not.
		//...

		//Return true if the solution has converged, otherwise false.
		if(hasConverged)
			return true;
		else
			return false;
	}
```
The specific details of the self-consistency callback is up to the application developer to fill in, but the general structure has to be as above.
That is, the callback has to accept the DiagonalizationSolver as an argument, perform the required work, determine whether the solution has converged and return true or false depending on wheter it has or not.
In addition to the self-consistency callback, the application developer interested in developing such a self-consistent calculation should also make use of the HoppingAmplitude callback described in the Model chapter for passing relevant parameters back to the Model in the next iteration step.

Once a self-consistency callback is implemented, the DiagonalizationSolver can be configured as follows to make use of it
```cpp
	DiagonalizationSolver solver;
	solver.setModel(model);
	solver.setSelfConsistencyCallback(selfConsistencyCallback);
	solver.setMaxIterations(100);
	solver.run();
```
Here the third line tells the Solver which function to use as a callback, while the fourth line puts an upper limit to the number of self-consistent steps the Solver will take if self-consistency is not reached.


# BlockDiagonalizationSolver {#BlockDiagoanlizationSolver}
The BlockDiagonalizationSolver is similar to the DiagonalizationSolver, except that it take advantage of possible block-diagonal structures in the Hamiltonian.
For this to work it is important that the Models index structure is chosen such that TBTK is able to automatically detect the block-diagonal structure of the Hamiltonian, as described in the Model chapter.
The BlockDiagonalizationSolver mimics the DiagonalizationSolver almost perfectly, and the code for initializing and running a BlockDiagonalizationSolver including a self-consistency callback is
```cpp
	DiagonalizationSolver solver;
	solver.setModel(model);
	solver.setSelfConsistencyCallback(selfConsistencyCallback);
	solver.setMaxIterations(100);
	solver.run();	
```
The only difference here is that the *selfCOnsistencyCallback* has to take a BlockDiagonalizationSolver as argument rather than a DiagonalizationSolver.

# ArnoldiSolver {#ArnoldiSolver}

# ChebyshevSolver {#ChebyshevSolver}

@page PropertyExtractors PropertyExtractors
