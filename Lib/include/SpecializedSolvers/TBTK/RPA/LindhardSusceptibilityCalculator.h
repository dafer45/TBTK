/* Copyright 2017 Kristofer Björnson
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

#ifndef COM_DAFER45_TBTK_LINDHARD_SUSCEPTIBILITY_CALCULATOR
#define COM_DAFER45_TBTK_LINDHARD_SUSCEPTIBILITY_CALCULATOR

#include "TBTK/RPA/MomentumSpaceContext.h"
#include "TBTK/RPA/SusceptibilityCalculator.h"

#include <complex>

#include <omp.h>

namespace TBTK{

class LindhardSusceptibilityCalculator : public SusceptibilityCalculator{
public:
	/** Constructor. */
	LindhardSusceptibilityCalculator(
		const MomentumSpaceContext &momentumSpaceContext
	);

	/** Destructor. */
	virtual ~LindhardSusceptibilityCalculator();

	/** Create slave SusceptibilityCalcuator. The slave reuses internal
	 *  lookup tables used to speed up the calculations and should not be
	 *  used after the generating master have been destructed. */
	virtual LindhardSusceptibilityCalculator* createSlave();

	/** Calculate the susceptibility. */
	virtual std::complex<double> calculateSusceptibility(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices,
		std::complex<double> energy
	);

	/** Calculate the susceptibility. */
	virtual std::vector<std::complex<double>> calculateSusceptibility(
		const DualIndex &kDual,
		const std::vector<int> &orbitalIndices
	);

	/** Precompute susceptibilities. Will calculate the susceptibility for
	 *  all values using a parallel algorithm. Can speed up calculations if
	 *  most of the susceptibilities are needed. */
	void precompute(unsigned int numWorkers = 129);

	/** Set to true if the susceptibility is known to only be
	 *  evaluated at points away from poles. */
	void setSusceptibilityIsSafeFromPoles(
		bool susceptibilityIsSafeFromPoles
	);

	/** Get whether the susceptibility is known to only be
	 *  evaluated at points away from poles. */
	bool getSusceptibilityIsSafeFromPoles() const;
private:
	/** Flag indicating whether the susceptibility is known to only
	 *  be evaluated at points away from poles. */
	bool susceptibilityIsSafeFromPoles;

	/** Fermi-Dirac distribution lookup table. */
	double *fermiDiracLookupTable;

	/** Slave constructor. */
	LindhardSusceptibilityCalculator(
		const MomentumSpaceContext &momentumSpaceContext,
		int *kPlusQLookupTable,
		double *fermiDiracLookupTable
	);

	/** Calculate the susceptibility using the Lindhard function. */
	std::complex<double> calculateSusceptibilityLindhard(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices,
		std::complex<double> energy
	);

	/** Calculate the susceptibility using the Lindhard function. */
	template<bool useKPlusQLookupTable, bool isSafeFromPoles>
	std::vector<std::complex<double>> calculateSusceptibilityLindhard(
		const DualIndex &kDual,
		const std::vector<int> &orbitalIndices
	);

	/** Get polt times two Fermi functions for use in the Linhard
	 *  function. */
	std::complex<double> getPoleTimesTwoFermi(
		std::complex<double> energy,
		double e2,
		double e1,
		double chemicalPotential,
		double temperature,
		int kPlusQLinearIndex,
		unsigned int meshPoint,
		unsigned int state2,
		unsigned int state1,
		unsigned int numOrbitals
	) const;
};

inline void LindhardSusceptibilityCalculator::setSusceptibilityIsSafeFromPoles(
	bool susceptibilityIsSafeFromPoles
){
	this->susceptibilityIsSafeFromPoles = susceptibilityIsSafeFromPoles;
}

inline bool LindhardSusceptibilityCalculator::getSusceptibilityIsSafeFromPoles() const{
	return susceptibilityIsSafeFromPoles;
}

};	//End of namespace TBTK

#endif
