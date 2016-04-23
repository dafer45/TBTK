#!/bin/bash -le

printf "\nRunning test\n"
templates=(BasicChebyshev
		BasicDiagonalization
		CarbonNanotube
		HexagonalLattice
		PartialBilayer
		SelfConsistentSuperconductivity
		SelfConsistentSuperconductivityChebyshev
		SelfConsistentSuperconductivityChebyshevWithRestrictedEnvironment
		TopologicalInsulator3D
		WireOnSuperconductor
		)

cd Templates
	for i in ${templates[@]}
	do
		cd $i
			printf "\nCompiling "${i}"\n"
			make
		cd ..
	done
cd ..

printf "\nAll templates successfully compiled\n\n"
