Examples {#Examples}
======

# Examples

## Basic models
- @subpage Chain1D
- @subpage SquareLattice2D
- @subpage CubicLattice3D
- @subpage AndersonDisorder

## Magnetisim
- @subpage Magnetism
- @subpage MagnetismSkyrmion

## Superconductivity
- @subpage Superconductivity
- @subpage SuperconductingVortex
- @subpage SuperconductivityMagneticImpurity

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

@page AndersonDisorder Anderson disorder
# Hamiltonian {#AndersonDisorderHamiltonian}
<center>\f$H = \sum_{\mathbf{i}}\left(U(\mathbf{i}) - \mu\right)c_{\mathbf{i}}^{\dagger}c_{\mathbf{i}} + t\sum_{\langle \mathbf{i}\mathbf{j}\rangle}c_{\mathbf{i}}^{\dagger}c_{\mathbf{j}}\f$</center>

# Code {#AndersonDisorderCode}
\snippet Examples/AndersonDisorder.cpp AndersonDisorder

# Output {#AndersonDisorderOutput}
\image html output/Examples/AndersonDisorder/figures/ExamplesAndersonDisorderDOS.png
\image html output/Examples/AndersonDisorder/figures/ExamplesAndersonDisorderDensity.png

@page Magnetism Magnetism
# Hamiltonian {#MagnetismHamiltonian}
<center>\f$H = -\mu\sum_{\mathbf{i}\sigma}c_{\mathbf{i}\sigma}^{\dagger}c_{\mathbf{i}\sigma} + t\sum_{\langle\mathbf{i}\mathbf{j}\rangle\sigma}c_{\mathbf{i}\sigma}^{\dagger}c_{\mathbf{j}\sigma} + J\sum_{\mathbf{i}\sigma}\left(\sigma_z\right)_{\sigma\sigma}c_{\mathbf{i}\sigma}^{\dagger}c_{\mathbf{i}\sigma}\f$</center>

# Code {#MagnetismCode}
\snippet Examples/Magnetism.cpp Magnetism

# Output {#MagnetismOutput}
\image html output/Examples/Magnetism/figures/ExamplesMagnetismDOS.png
\image html output/Examples/Magnetism/figures/ExamplesMagnetismMagnetization.png

@page MagnetismSkyrmion Skyrmion
# Hamiltonian {#MagnetismSkyrmionHamiltonian}
<center>\f$H = -\mu\sum_{\mathbf{i}\sigma}c_{\mathbf{i}\sigma}^{\dagger}c_{\mathbf{i}\sigma} + t\sum_{\langle\mathbf{i}\mathbf{j}\rangle\sigma}c_{\mathbf{i}\sigma}^{\dagger}c_{\mathbf{j}\sigma} + J\sum_{\mathbf{i}\sigma\sigma'}\left(\mathbf{S}(\mathbf{i})\cdot\boldsymbol{\sigma}\right)_{\sigma\sigma'}c_{\mathbf{i}\sigma}^{\dagger}c_{\mathbf{i}\sigma'}\f$</center>

# Code {#MagnetismSkyrmionCode}
\snippet Examples/MagnetismSkyrmion.cpp MagnetismSkyrmion

# Output {#MagnetismSkyrmionOutput}
\image html output/Examples/MagnetismSkyrmion/figures/ExamplesMagnetismSkyrmionMagnetizationX.png
\image html output/Examples/MagnetismSkyrmion/figures/ExamplesMagnetismSkyrmionMagnetizationY.png
\image html output/Examples/MagnetismSkyrmion/figures/ExamplesMagnetismSkyrmionMagnetizationZ.png

@page Superconductivity Superconductivity
# Hamiltonian {#SuperconductivityHamiltonian}
<center>\f$H = -\mu\sum_{\mathbf{i}}\left(c_{\mathbf{i}\uparrow}^{\dagger}c_{\mathbf{i}\uparrow} - c_{\mathbf{i}\downarrow}c_{\mathbf{i}\downarrow}^{\dagger}\right) + t\sum_{\langle\mathbf{i}\mathbf{j}\rangle}\left(c_{\mathbf{i}\uparrow}^{\dagger}c_{\mathbf{j}\uparrow} - c_{\mathbf{i}\downarrow}c_{\mathbf{j}\downarrow}^{\dagger}\right) + \sum_{\mathbf{i}}\left(\Delta c_{\mathbf{i}\uparrow}^{\dagger}c_{\mathbf{i}\downarrow}^{\dagger} + \Delta^{*}c_{\mathbf{i}\downarrow}c_{\mathbf{i}\uparrow}\right)\f$</center>

# Note
For ordinary s-wave superconductivity, only spin-up electrons and spin-down holes are considered.
This is why the Hamiltonian above has up-spin indices on the electron operators (creation if it is to the left and annihilation if it is to the right) and down-spin indices on the hole operators (annihilation if it is to the left and creation if it is to the right).

By introducing the notation \f$a_{\mathbf{i}0} = c_{\mathbf{i}\uparrow}\f$ and \f$a_{\mathbf{i}1} = c_{\mathbf{i}\downarrow}^{\dagger}\f$, we can rewrite the Hamiltonian as
<center>\f$H = -\mu\sum_{\mathbf{i}}\left(a_{\mathbf{i}0}^{\dagger}a_{\mathbf{i}0} - a_{\mathbf{i}1}^{\dagger}a_{\mathbf{i}1}\right) + t\sum_{\langle\mathbf{i}\mathbf{j}\rangle}\left(a_{\mathbf{i}0}^{\dagger}a_{\mathbf{j}0} - a_{\mathbf{i}1}^{\dagger}a_{\mathbf{j}1}\right) + \sum_{\mathbf{i}}\left(\Delta a_{\mathbf{i}0}^{\dagger}a_{\mathbf{i}1} + \Delta^{*}a_{\mathbf{i}1}^{\dagger}a_{\mathbf{i}0}\right)\f$</center>
This is on the same form as an ordinary bilinear Hamiltonian and therefore allows us to solve the problem as such.

# Code {#SuperconductivityCode}
\snippet Examples/Superconductivity.cpp Superconductivity

# Output {#SuperconductivityOutput}
\image html output/Examples/Superconductivity/figures/ExamplesSuperconductivityDOS.png

@page SuperconductingVortex Superconducting vortex (Caroli-de Gennes-Matricon)
# Hamiltonian {#SuperconductingVortexHamiltonian}
<center>\f$H = -\mu\sum_{\mathbf{i}}\left(c_{\mathbf{i}\uparrow}^{\dagger}c_{\mathbf{i}\uparrow} - c_{\mathbf{i}\downarrow}c_{\mathbf{i}\downarrow}^{\dagger}\right) + t\sum_{\langle\mathbf{i}\mathbf{j}\rangle}\left(c_{\mathbf{i}\uparrow}^{\dagger}c_{\mathbf{j}\uparrow} - c_{\mathbf{i}\downarrow}c_{\mathbf{j}\downarrow}^{\dagger}\right) + \sum_{\mathbf{i}}\left(\Delta(\mathbf{i}) c_{\mathbf{i}\uparrow}^{\dagger}c_{\mathbf{i}\downarrow}^{\dagger} + \Delta^{*}(\mathbf{i})c_{\mathbf{i}\downarrow}c_{\mathbf{i}\uparrow}\right)\f$</center>

# Code {#SuperconductingVortexCode}
\snippet Examples/SuperconductingVortex.cpp SuperconductingVortex

# Output {#SuperconductingVortexOutput}
\image html output/Examples/SuperconductingVortex/figures/ExamplesSuperconductingVortexLDOS.png

@page SuperconductivityMagneticImpurity Magnetic impurity (Yu-Shiba-Rusinov)
# Hamiltonian {#SuperconductivityMagneticImpurityHamiltonian}
<center>\f{eqnarray*}{
	H &=& \sum_{\mathbf{i}\sigma}\left(\left(\delta_{\mathbf{i}\mathbf{I}}J\left(\sigma_z\right)_{\sigma\sigma} - \mu\right)c_{\mathbf{i}\sigma}^{\dagger}c_{\mathbf{i}\sigma} - \left(\delta_{\mathbf{i}\mathbf{I}}J\left(\sigma_{z}\right)_{\sigma\sigma} - \mu\right)c_{\mathbf{i}\sigma}c_{\mathbf{i}\sigma}^{\dagger}\right)\\
	&+& t\sum_{\langle\mathbf{i}\mathbf{j}\rangle\sigma}\left(c_{\mathbf{i}\sigma}^{\dagger}c_{\mathbf{j}\sigma} - c_{\mathbf{i}\sigma}c_{\mathbf{j}\sigma}^{\dagger}\right)\\
	&+& \sum_{\mathbf{i}}\left(\Delta c_{\mathbf{i}\uparrow}c_{\mathbf{i}\downarrow} - \Delta c_{\mathbf{i}\downarrow}c_{\mathbf{i}\uparrow} + H.c.\right),
\f}</center>
where \f$\mathbf{I}\f$ is the impurity site.

# Code {#SuperconductivityMagneticImpurityCode}
\snippet Examples/SuperconductivityMagneticImpurity.cpp SuperconductivityMagneticImpurity

# Output {#SuperconductivityMagneticImpurityOutput}
\image html output/Examples/SuperconductivityMagneticImpurity/figures/ExamplesSuperconductivityMagneticImpurityLDOS.png
\image html output/Examples/SuperconductivityMagneticImpurity/figures/ExamplesSuperconductivityMagneticImpurityEigenValues.png

@page KitaevModel Kitaev Model
# Hamiltonian {#KitaevModelHamiltonian}
<center>\f$H = t\sum_{x}\left(c_{x+1}^{\dagger}c_{x} - c_{x+1}x_{x}^{\dagger}\right) + \frac{i}{10}\sum_{x}c_{x+1}c_{x} + H.c.\f$</center>

# Code {#KitaevModelCode}
\snippet Examples/KitaevModel.cpp KitaevModel

# Output {#KitaevModelOutput}
\image html output/Examples/KitaevModel/figures/ExamplesKitaevModelLDOS.png
\image html output/Examples/KitaevModel/figures/ExamplesKitaevModelWaveFunction.png
