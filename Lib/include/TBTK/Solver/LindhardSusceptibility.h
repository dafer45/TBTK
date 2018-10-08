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
		const RPA::MomentumSpaceContext &momentumSpaceContext
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
		const RPA::MomentumSpaceContext &momentumSpaceContext,
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

	/** Generate lookup table for the k+q linear index. Can be called
	 *  repeatedly, and the lookup table is only generated the first time.
	 */
	void generateKPlusQLookupTable();

	/** Returns the k+q lookup table. */
	int* getKPlusQLookupTable();

	/** Returns the k+q lookup table. */
	const int* getKPlusQLookupTable() const;

        /** Returns the linear index for k+q. */
	template<bool useKPlusQLookupTable>
	int getKPlusQLinearIndex(
		unsigned int meshIndex,
		const std::vector<double> &k,
		int kLinearIndex
	) const;

	/** Lookup table for calculating k+q. */
	int *kPlusQLookupTable;

	/** Flag indicating whether the LindhardSusceptibility is a master.
	 *  Masters own resources shared between masters and slaves and are
	 *  responsible for cleaning up. */
	bool isMaster;
};

inline int* LindhardSusceptibility::getKPlusQLookupTable(){
	return kPlusQLookupTable;
}

inline const int* LindhardSusceptibility::getKPlusQLookupTable() const{
	return kPlusQLookupTable;
}

template<>
inline int LindhardSusceptibility::getKPlusQLinearIndex<false>(
	unsigned int meshIndex,
	const std::vector<double> &k,
	int kLinearIndex
) const{
	const RPA::MomentumSpaceContext &momentumSpaceContext
		= getMomentumSpaceContext();

	const std::vector<std::vector<double>> &mesh
		= momentumSpaceContext.getMesh();

	Index kPlusQIndex
		= momentumSpaceContext.getBrillouinZone().getMinorCellIndex(
			{mesh[meshIndex][0] + k[0], mesh[meshIndex][1] + k[1]},
			momentumSpaceContext.getNumMeshPoints()
		);
	return momentumSpaceContext.getModel().getHoppingAmplitudeSet().getFirstIndexInBlock(
		kPlusQIndex
	);
}

template<>
inline int LindhardSusceptibility::getKPlusQLinearIndex<true>(
	unsigned int meshIndex,
	const std::vector<double> &k,
	int kLinearIndex
) const{
	const RPA::MomentumSpaceContext &momentumSpaceContext
		= getMomentumSpaceContext();

	return kPlusQLookupTable[
		meshIndex*momentumSpaceContext.getMesh().size()
		+ kLinearIndex/momentumSpaceContext.getNumOrbitals()
	];
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
