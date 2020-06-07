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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file SelfEnergy.h
 *  @brief Calculates the self-energy.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SOLVER_SELF_ENERGY
#define COM_DAFER45_TBTK_SOLVER_SELF_ENERGY

#include "TBTK/BrillouinZone.h"
#include "TBTK/IndexedDataTree.h"
#include "TBTK/Property/InteractionVertex.h"
#include "TBTK/RPA/MomentumSpaceContext.h"
#include "TBTK/Solver/Solver.h"

namespace TBTK{
namespace Solver{

class SelfEnergy : public Solver{
	TBTK_DYNAMIC_TYPE_INFORMATION(SelfEnergy)
public:
	/** Constructor. */
	SelfEnergy(
		const RPA::MomentumSpaceContext &momentumSpaceContext,
		const Property::InteractionVertex &interactionVertex
	);

	/** Destructor. */
	~SelfEnergy();

	/** Get momentum cpsace context. */
	const RPA::MomentumSpaceContext& getMomentumSpaceContext() const;

	/** Get the InteractionVertex.
	 *
	 *  @return The InteractionVertex. */
	const Property::InteractionVertex& getInteractionVertex() const;

	/** Initialize the SelfEnergyCalculator. */
	void init();

	/** Calculate self-energy. */
	std::vector<std::complex<double>> calculateSelfEnergy(
		const Index &index,
		const std::vector<std::complex<double>> &energies
	);

	/** Calculate self-energy self-consistently. */
	std::vector<std::complex<double>> calculateSelfEnergySelfConsistently(
		unsigned int numMatsubaraFrequencies,
		const std::vector<std::complex<double>> &energies
	);
private:
	/** Momentum space context. */
	const RPA::MomentumSpaceContext &momentumSpaceContext;

	/** Interaction vertex. */
	const Property::InteractionVertex &interactionVertex;

	/** Lookup table for calculating k+q. */
	int *kMinusQLookupTable;

	/** Generate lookup table for the k-q linear index. Can be called
	 *  repeatedly, and the lookup table is only generated once. */
	void generateKMinusQLookupTable();

	/** Returns the linear index for k+q. */
	template<bool useKPlusKLookupTable>
	int getKMinusQLinearIndex(
		unsigned int meshIndex,
		const std::vector<double> &k,
		int kLinearIndex
	) const;

	/** Flag indicating whether the SelfEnergyCalculator is initialized. */
	bool isInitialized;

	/** Self-energy main loop. */
	template<bool singleSelfEnergyEnergy>
	void selfEnergyMainLoop(
		const Index &index,
		const std::vector<std::complex<double>> &energies,
		std::vector<std::complex<double>> &result
	);
};

inline const RPA::MomentumSpaceContext& SelfEnergy::getMomentumSpaceContext(
) const{
	return momentumSpaceContext;
}

inline const Property::InteractionVertex& SelfEnergy::getInteractionVertex(
) const{
	return interactionVertex;
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
/// @endcond
