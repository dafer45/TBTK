Examples {#Examples}
======

# Examples

## Basic models
- @subpage Chain1D
- @subpage SquareLattice2D
- @subpage CubicLattice3D

## Subperconductivity
- @subpage Superconductivity

## Topological superconductivity
- @subpage KitaevModel

@page Chain1D 1D chain
# Hamiltonian {#Chain1DHamiltonian}
<center>\f$H = -\mu\sum_{x}c_{x}^{\dagger}c_{x} + t\sum_{x}c_{x+1}^{\dagger}c_{x}\f$</center>

# Code {#Chain1DCode}
\snippet Examples/Chain1D.cpp Chain1D

# Output {#Chain1DOutput}
\image html output/Examples/Chain1D/figures/ExamplesChain1DDOS.png
\image html output/Examples/Chain1D/figures/ExamplesChain1DWaveFunctions.png

@page SquareLattice2D 2D square lattice
# Hamiltonian {#SquareLattice2DHamiltonian}
<center>\f$H = -\mu\sum_{\mathbf{i}}c_{\mathbf{i}}^{\dagger}c_{\mathbf{i}} + t\sum_{\langle \mathbf{i}\mathbf{j}\rangle}c_{\mathbf{i}}^{\dagger}c_{\mathbf{j}}\f$</center>

# Code {#SquareLattice2DCode}
\snippet Examples/SquareLattice2D.cpp SquareLattice2D

# Output {#SquareLattice2DOutput}
\image html output/Examples/SquareLattice2D/figures/ExamplesSquareLattice2DDOS.png
\image html output/Examples/SquareLattice2D/figures/ExamplesSquareLattice2DWaveFunction.png

@page CubicLattice3D 3D cubic lattice
# Hamiltonian {#CubicLattice3DHamiltonian}
<center>\f$H = -\mu\sum_{\mathbf{i}}c_{\mathbf{i}}^{\dagger}c_{\mathbf{i}} + t\sum_{\langle \mathbf{i}\mathbf{j}\rangle}c_{\mathbf{i}}^{\dagger}c_{\mathbf{j}}\f$</center>

# Code {#CubicLattice3DCode}
\snippet Examples/CubicLattice3D.cpp CubicLattice3D

# Output {#CubicLattice3DOutput}
\image html output/Examples/CubicLattice3D/figures/ExamplesCubicLattice3DDOS.png
\image html output/Examples/CubicLattice3D/figures/ExamplesCubicLattice3DWaveFunction.png

@page Superconductivity Superconductivity
# Hamiltonian {#SuperconductivityHamiltonian}
<center>\f$H = -\mu\sum_{\mathbf{i}}\left(c_{\mathbf{i}}^{\dagger}c_{\mathbf{i}} - c_{\mathbf{i}}c_{\mathbf{i}}^{\dagger}\right) + t\sum_{\langle\mathbf{i}\mathbf{j}\rangle}\left(c_{\mathbf{i}}^{\dagger}c_{\mathbf{j}} - c_{\mathbf{i}}c_{\mathbf{j}}^{\dagger}\right) + \sum_{\mathbf{i}}\left(\Delta c_{\mathbf{i}}c_{\mathbf{i}} + \Delta^{*}c_{\mathbf{i}}^{\dagger}c_{\mathbf{i}}^{\dagger}\right)\f$</center>

# Code {#SuperconductivityCode}
\snippet Examples/Superconductivity.cpp Superconductivity

# Output {#SuperconductivityOutput}
\image html output/Examples/Superconductivity/figures/ExamplesSuperconductivityDOS.png

@page KitaevModel Kitaev Model
# Hamiltonian {#KitaevModelHamiltonian}
<center>\f$H = t\sum_{x}\left(c_{x+1}^{\dagger}c_{x} - c_{x+1}x_{x}^{\dagger}\right) + \frac{i}{10}\sum_{x}c_{x+1}c_{x} + H.c.\f$</center>

# Code {#KitaevModelCode}
\snippet Examples/KitaevModel.cpp KitaevModel

# Output {#KitaevModelOutput}
\image html output/Examples/KitaevModel/figures/ExamplesKitaevModelLDOS.png
\image html output/Examples/KitaevModel/figures/ExamplesKitaevModelWaveFunction.png
