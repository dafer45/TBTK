# Start Page {#mainpage}

\image html Logo.png"

# Welcome to the documentation for TBTK!
TBTK is an open-source C++ framework for modeling and solving problems formulated using the language of second quantization.
It can be used to set up general models with little effort and provides a variety of native solution methods.

To get started, see the [installation instructions](@ref InstallationInstructions), [manual](@ref Manual), [examples](@ref Examples), and [tutorials](@ref Tutorials).
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

# Examples
<table border="0">
	<tr>
		<td>
			[Superconductivity](@ref Superconductivity)
			<img src="ExamplesSuperconductivityDOS.png" style="width:250px" />
		</tr>
		<td>
			[Caroli-de Gennes-Matricon](@ref SuperconductingVortex)
			<img src="ExamplesSuperconductingVortexLDOS.png" style="width:250px" />
		</td>
		<td>
			[Magnetism](@ref Magnetism)
			<img src="ExamplesMagnetismDOS.png" style="width:250px" />
		</tr>
	</tr>
	<tr>
		<td>
			[Kitaev model](@ref KitaevModel)
			<img src="ExamplesKitaevModelLDOS.png" style="width:250px" />
		</td>
		<td>
			[Anderson disorder](@ref AndersonDisorder)
			<img src="ExamplesAndersonDisorderDOS.png" style="width:250px" />
		</td>
		<td>
			[Yu-Shiba-Rusinov](@ref SuperconductivityMagneticImpurity)
			<img src="ExamplesSuperconductivityMagneticImpurityLDOS.png" style="width:250px" />
		</td>
	</tr>
</table>
