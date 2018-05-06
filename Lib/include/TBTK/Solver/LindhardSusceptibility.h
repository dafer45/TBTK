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
 *  @file LindhardSuscesptibility.h
 *  @brief Property container for density
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SOLVER_LINDHARD_SUSCEPTIBILITY
#define COM_DAFER45_TBTK_SOLVER_LINDHARD_SUSCEPTIBILITY

#include "TBTK/RPA/MomentumSpaceContext.h"
#include "TBTK/Solver/Susceptibility.h"

#include <complex>

namespace TBTK{
namespace Solver{

class LindhardSusceptibility : public Susceptibility{
public:
	/** Constructor. */
	LindhardSusceptibility(
		const MomentumSpaceContext &momentumSpaceContext
	);

	/** Destructor. */
	virtual ~LindhardSusceptibility();

	/** Create slave SusceptibilityCalcuator. The slave reuses internal
	 *  lookup tables used to speed up the calculations and should not be
	 *  used after the generating master have been destructed. */
	virtual LindhardSusceptibility* createSlave();

	/** Calculate the susceptibility. */
	virtual std::vector<std::complex<double>> calculateSusceptibility(
		const Index &index,
		const std::vector<std::complex<double>> &energies
	);
private:
	/** Fermi-Dirac distribution lookup table. */
	double *fermiDiracLookupTable;

	/** Slave constructor. */
	LindhardSusceptibility(
		const MomentumSpaceContext &momentumSpaceContext,
		int *kPlusQLookupTable,
		double *fermiDiracLookupTable
	);

	/** Calculate the susceptibility using the Lindhard function. */
	template<bool useKPlusQLookupTable/*, bool isSafeFromPoles*/>
	std::vector<std::complex<double>> calculateSusceptibilityLindhard(
		const Index &index,
		const std::vector<std::complex<double>> &energies
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

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
