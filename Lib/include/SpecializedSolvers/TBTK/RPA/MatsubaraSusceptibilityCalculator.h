/* Copyright 2018 Kristofer Björnson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @package TBTKcalc
 *  @file SuscesptibilityCalculator.h
 *  @brief Property container for density
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_MATSUBARA_SUSCEPTIBILITY_CALCULATOR
#define COM_DAFER45_TBTK_MATSUBARA_SUSCEPTIBILITY_CALCULATOR

#include "TBTK/RPA/SusceptibilityCalculator.h"

#include <complex>

//#include <omp.h>

namespace TBTK{

class MatsubaraSusceptibilityCalculator : public SusceptibilityCalculator{
public:
	/** Constructor. */
	MatsubaraSusceptibilityCalculator(
		const MomentumSpaceContext &momentumSpaceContext
	);

	/** Destructor. */
	~MatsubaraSusceptibilityCalculator();

	/** Create slave SusceptibilityCalcuator. The slave reuses internal
	 *  lookup tables used to speed up the calculations and should not be
	 *  used after the generating master have been destructed. */
	MatsubaraSusceptibilityCalculator* createSlave();

	/** Calculate the susceptibility. */
	std::complex<double> calculateSusceptibility(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices,
		std::complex<double> energy
	);

	/** Calculate the susceptibility. */
	std::vector<std::complex<double>> calculateSusceptibility(
		const DualIndex &kDual,
		const std::vector<int> &orbitalIndices
	);

	/** Set the number of summation energies. */
	void setNumSummationEnergies(unsigned int numSummationEnergies);
private:
	/** Green's function for use in Mode::Matsubara. */
	std::complex<double> *greensFunction;

	/** Summation energies. Used in Mode::Matsubara. */
	std::vector<std::complex<double>> summationEnergies;

	/** Slave constructor. */
	MatsubaraSusceptibilityCalculator(
		const MomentumSpaceContext &momentumSpaceContext,
		int *kPlusQLookupTable
	);

	/** Calculate the susceptibility using the Matsubara sum. */
	std::complex<double> calculateSusceptibilityMatsubara(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices,
		std::complex<double> energy
	);

	/** Calculate the susceptibility using the Matsubara sum. */
	template<bool useKPlusQLookupTable>
	std::vector<std::complex<double>> calculateSusceptibilityMatsubara(
		const DualIndex &kDual,
		const std::vector<int> &orbitalIndices
	);

	/** Calculate Green's function. */
	void calculateGreensFunction();

	/** Get greensFunctionValue. */
	std::complex<double>& getGreensFunctionValue(
		unsigned int meshPoint,
		unsigned int orbital0,
		unsigned int orbital1,
		unsigned int energy,
		unsigned int numEnergies,
		unsigned int numOrbitals
	);
};

inline void MatsubaraSusceptibilityCalculator::setNumSummationEnergies(
	unsigned int numSummationEnergies
){
	TBTKAssert(
		numSummationEnergies%2 == 1,
		"MatsubaraSusceptibilityCalculator::setNumSummationEnergies()",
		"The number of summation energies must be an odd number.",
		""
	);

	double temperature = UnitHandler::convertTemperatureNtB(
		getMomentumSpaceContext().getModel().getTemperature()
	);
	double kT = UnitHandler::getK_BB()*temperature;
	double hbar = UnitHandler::getHbarB();

	summationEnergies.clear();
	for(unsigned int n = 0; n < numSummationEnergies; n++){
		summationEnergies.push_back(
			M_PI*(2*(n - numSummationEnergies/2))*kT/hbar
		);
	}

	clearCache();
	if(greensFunction != nullptr){
		delete [] greensFunction;
		greensFunction = nullptr;
	}
}

/*inline std::vector<std::complex<double>> SusceptibilityCalculator::calculateSusceptibility(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices
){
	return calculateSusceptibility(
		DualIndex(
			momentumSpaceContext->getKIndex(k),
			k
		),
		orbitalIndices
	);
}*/

inline std::complex<double>& MatsubaraSusceptibilityCalculator::getGreensFunctionValue(
	unsigned int meshPoint,
	unsigned int orbital0,
	unsigned int orbital1,
	unsigned int energy,
	unsigned int numEnergies,
	unsigned int numOrbitals
){
	return greensFunction[
		numEnergies*(
			numOrbitals*(numOrbitals*meshPoint + orbital0)
			+ orbital1
		) + energy
	];
}

};	//End of namespace TBTK

#endif
