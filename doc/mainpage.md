# Start Page {#mainpage}

# Welcome to the documentation for TBTK!
TBTK is an open-source C++ framework for modeling and solving problems formulated using the language of second quantization.
It can be used to set up general models with little effort and provides a variety of native solution methods.

To get started, see the [installation instructions](@ref InstallationInstructions), the [manual](@ref Manual), and the [tutorials](@ref Tutorials).
Also, see the blog posts and other resources collected on [second-tech.com](http://second-tech.com/wordpress/index.php/tbtk/).

## Download TBTK
Download TBTK from [GitHub](https://github.com/dafer45/TBTK).
See the [installation instructions](@ref InstallationInstructions) to make sure you checkout the right version before installation.

## Core strengths
- The speed of a low-level language with the syntax of a high-level language.
- Results in readable code that puts emphasis on the physics.
- Allows for a wide variety of models and solution methods to be combined in different ways.
- Focus on your own task, while still benefiting from the work of others.
- Gives method developers complete freedom to optimize their solvers without having to worry about model-specific details.
- A versioning system that ensures that results are reproducible forever.

## Native production-ready solvers
- @link SolverDiagonalizer Diagonalization@endlink
- @link SolverBlockDiagonalizer Block diagonalization@endlink
- @link SolverArnoldiIterator Arnoldi iteration@endlink
- @link SolverChebyshevExpander Chebyshev expansion@endlink

# Example
## Problem formulation
Consider a two-dimensional substrate of size 51x51 described by the Hamiltonian
<center>\f$ H_{S} = U_S\sum_{\mathbf{i}\sigma}c_{\mathbf{i}\sigma}^{\dagger}c_{\mathbf{i}\sigma} - t\sum_{\langle\mathbf{i}\mathbf{j}\rangle\sigma}c_{\mathbf{i}\sigma}^{\dagger}c_{\mathbf{j}\sigma}\f$.</center>
Here \f$\mathbf{i}\f$ is a two-dimensional index, \f$\sigma\f$ is a spin index, and \f$\langle\mathbf{i}\mathbf{j}\rangle\f$ denotes summation over nearest-neighbors.
Further, consider a magnetic impurity on top of the substrate described by the Hamiltonian
<center>\f$ H_{Imp} = (U_{Imp} - J)d_{\uparrow}^{\dagger}d_{\uparrow} + (U_{Imp} + J)d_{\downarrow}^{\dagger}d_{\downarrow}.\f$</center>
Finally, the impurity connects to the site (25, 25) in the substrate through the term
<center>\f$ H_{Int} = \delta\sum_{\sigma}c_{(25,25)\sigma}^{\dagger}d_{\sigma} + H.c.\f$</center>

The total Hamiltonian is
<center>\f$H = H_{S} + H_{Imp} + H_{Int}\f$.</center>

<!--<img src="MainPageModel.png" style="max-width: 800px" />  
<i>A magnetic impurity on top of a square lattice. Image generated using the built in RayTracer (currently in experimental development stage).</i>  -->

<b>Question:</b> What is the spin-polarized LDOS in the substrate as a function of \f$U_S, U_{Imp}, t, J\f$, and \f$\delta\f$?

## Numerical solution
```cpp
	//Parameters.
	const int SIZE_X = 51;
	const int SIZE_Y = 51;

	double U_S   = 1;
	double U_Imp = 1;
	double t     = 1;
	double J     = 1;
	double delta = 1;

	//Create model.
	Model model;

	//Setup substrate.
	for(int x = 0; x < SIZE_X; x++){
		for(int y = 0; y < SIZE_Y; y++){
			for(int spin = 0; spin < 2; spin++){
				model << HoppingAmplitude(
					U_S,
					{0, x, y, spin},
					{0, x, y, spin}
				);

				if(x+1 < SIZE_X){
					model << HoppingAmplitude(
						-t,
						{0, x+1, y, spin},
						{0, x,   y, spin}
					) + HC;
				}
				if(y+1 < SIZE_Y){
					model << HoppingAmplitude(
						-t,
						{0, x, y+1, spin},
						{0, x, y,   spin}
					) + HC;
				}
			}
		}
	}

	for(int s = 0; s < 2; s++){
		//Setup impurity.
		model << HoppingAmplitude(        U_Imp, {1, spin}, {1, spin});
		model << HoppingAmplitude(-J*(1-2*spin), {1, spin}, {1, spin});

		//Add coupling between the substrate and impurity.
		model << HoppingAmplitude(
			delta,
			{0, SIZE_X/2, SIZE_Y/2, spin},
			{1, spin}
		) + HC;
	}

	//Construct model.
	model.construct();

	//Setup Solver
	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	//Extract the spin-polarized LDOS in the substrate.
	PropertyExtractor::Diagonalizer propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(-1, 1, 1000);
	Property::SpinPolarizedLDOS spinPolarizedLDOS
		= propertyExtractor.calculateSpinPolarizedLDOS(
			{{0, _a_, _a_, IDX_SPIN}}
		);
```
