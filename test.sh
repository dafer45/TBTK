#!/bin/bash -le

echo "#################"
echo "# Running tests #"
echo "#################"
templates=(BasicChebyshev
		BasicDiagonalization
		CarbonNanotube
		HexagonalLattice
		PartialBilayer
		SelfConsistentSuperconductivity
		SelfConsistentSuperconductivityChebyshev
		SelfConsistentSuperconductivityChebyshevWithRestrictedEnvironment
#		TopologicalInsulator3D
		WireOnSuperconductor
		)

cd Templates
	for i in ${templates[@]}
	do
		cd $i
			make
		cd ..
	done
cd ..

printf "\nAll templates successfully built.\n\n"
