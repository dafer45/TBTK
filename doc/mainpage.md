# Start Page {#mainpage}

TBTK is a C++ library for modeling and solving second quantized Hamiltonians.
A flexible indexing scheme allows for general models to be setup with little effort, and a variety of solution methods are implemented in natively available solvers.
The aim is to provide flexible and well thought through data structures that combines the efficiency of C++ with high level abstraction that allows developers to focus their attention on physics rather than numerics.
Thereby facilitating both the investigation of specific physical question as well as enabling rapid development of completely new methods.

To get started, see the [installation instructions](@ref InstallationInstructions), the [manual](@ref Manual), and the [tutorials](@ref Tutorials).

The code is available at https://github.com/dafer45/TBTK.

# Example
## Problem formulation

<img src="MainPageModel.png" style="max-width: 800px" />  
<i>A magnetic impurity on top of a square lattice. Image generated using the built in RayTracer (currently in experimental development stage).</i>  

Consider a magnetic impurity on top of a two-dimensional substrate of size 51x51 described by
<center>\f$H = H_{S} + H_{Imp} + H_{Int}\f$</center>
where
<center>\f{eqnarray*}{
	H_{S} &=& U_S\sum_{\mathbf{i}\sigma}c_{\mathbf{i}\sigma}^{\dagger}c_{\mathbf{i}\sigma} - t\sum_{\langle\mathbf{i}\mathbf{j}\rangle\sigma}c_{\mathbf{i}\sigma}^{\dagger}c_{\mathbf{j}\sigma},\\
	H_{Imp} &=& (U_{Imp} - J)d_{\uparrow}^{\dagger}d_{\uparrow} + (U_{Imp} + J)d_{\downarrow}^{\dagger}d_{\downarrow},\\
	H_{Int} &=& \delta\sum_{\sigma}c_{(25,25)\sigma}^{\dagger}d_{\sigma} + H.c.
\f}</center>
Here \f$\mathbf{i}\f$ is a two-dimensional index, \f$\langle\mathbf{i}\mathbf{j}\rangle\f$ indicates summation over nearest neighbors, \f$\sigma\f$ is a spin index, and \f$c_{\mathbf{i}}\f$ and \f$d_{\sigma}\f$ are operators on the substrate and impurity, respectively.
The parameters \f$U_S\f$ and \f$U_{Imp}\f$ are onsite energies on the substrate and impurity, respectively, while \f$t\f$ is a nearest neighbor hopping amplitude, \f$J\f$ is a Zeeman term, and \f$\delta\f$ is the coupling strength between the substrate and impurity.

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
					model << HoppingAmplitude(-t,
						{0, x, y+1, s},
						{0, x, y,   s}
					) + HC;
				}
			}
		}
	}

	for(int s = 0; s < 2; s++){
		//Setup impurity.
		model << HoppingAmplitude(     U_Imp, {1, s}, {1, s});
		model << HoppingAmplitude(-J*(1-2*s), {1, s}, {1, s});

		//Add coupling between the substrate and impurity.
		model << HoppingAmplitude(
			delta,
			{0, SIZE_X/2, SIZE_Y/2, s},
			{1, s}
		) + HC;
	}

	//Construct model.
	model.construct();

	//Setup Solver
	Solver::Diagonalizer solver;
	solver.setModel(model);
	solver.run();

	//Extract the spin-polarized LDOS.
	PropertyExtractor::Diagonalizer propertyExtractor(solver);
	propertyExtractor.setEnergyWindow(-1, 1, 1000);
	Property::SpinPolarizedLDOS spinPolarizedLDOS
		= propertyExtractor.calculateSpinPolarizedLDOS(
			{{0, _a_, _a_, IDX_SPIN}}
		);

	//Save result.
	FileWriter::writeSpinPolarizedLDOS(spinPolarizedLDOS);
```
