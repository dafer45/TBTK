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
 *  @file SelfEnergy2.h
 *  @brief Calculates the self-energy.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SOLVER_SELF_ENERGY2
#define COM_DAFER45_TBTK_SOLVER_SELF_ENERGY2

#include "TBTK/Property/GreensFunction.h"
#include "TBTK/Property/InteractionVertex.h"
#include "TBTK/Property/SelfEnergy.h"
#include "TBTK/MomentumSpaceContext.h"
#include "TBTK/Solver/Solver.h"

namespace TBTK{
namespace Solver{

class SelfEnergy2 : public Solver, public Communicator{
public:
	/** Constructor. */
	SelfEnergy2(
		const MomentumSpaceContext &momentumSpaceContext,
		const Property::InteractionVertex &interactionVertex,
		const Property::GreensFunction &greensFunction
	);

	/** Get momentum cpsace context. */
	const MomentumSpaceContext& getMomentumSpaceContext() const;

	/** Get the InteractionVertex.
	 *
	 *  @return The InteractionVertex. */
	const Property::InteractionVertex& getInteractionVertex() const;

	/** Get the GreensFunction.
	 *
	 *  @return The GreensFunction. */
	const Property::GreensFunction& getGreensFunction() const;

	/** Calculate the self-energy. */
	std::vector<std::complex<double>> calculateSelfEnergy(
		const Index &index,
		int lowerMatsubaraEnergyIndex,
		int upperMatsubaraEnergyIndex
	);

	/** Calculate the self-energy. */
	Property::SelfEnergy calculateSelfEnergyAllBlocks(
		const Index &index,
		int lowerMatsubaraEnergyIndex,
		int upperMatsubaraEnergyIndex
	);
private:
	/** Momentum space context. */
	const MomentumSpaceContext &momentumSpaceContext;

	/** Interaction vertex. */
	const Property::InteractionVertex &interactionVertex;

	/** GreensFunction. */
	const Property::GreensFunction &greensFunction;

	/** Get the number of intra block Indices. */
	unsigned int getNumIntraBlockIndices();

	/** Get a list of intra block Indices. */
	std::vector<Index> getIntraBlockIndexList();
};

inline const MomentumSpaceContext& SelfEnergy2::getMomentumSpaceContext(
) const{
	return momentumSpaceContext;
}

inline const Property::InteractionVertex& SelfEnergy2::getInteractionVertex(
) const{
	return interactionVertex;
}

inline const Property::GreensFunction& SelfEnergy2::getGreensFunction(
) const{
	return greensFunction;
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
