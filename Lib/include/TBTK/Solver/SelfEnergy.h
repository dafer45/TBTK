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
public:
	/** Constructor. */
	SelfEnergy(
		const MomentumSpaceContext &momentumSpaceContext,
		const Property::InteractionVertex &interactionVertex
	);

	/** Destructor. */
	~SelfEnergy();

	/** Get momentum cpsace context. */
	const MomentumSpaceContext& getMomentumSpaceContext() const;

	/** Get the InteractionVertex.
	 *
	 *  @return The InteractionVertex. */
	const Property::InteractionVertex& getInteractionVertex() const;

	/** Initialize the SelfEnergyCalculator. */
	void init();

	/** Set the energies for which the self-energy should be
	 *  calculated. */
/*	void setSelfEnergyEnergies(
		const std::vector<std::complex<double>> &selfEnergyEnergies
	);*/

	/** Calculate self-energy. */
/*	std::vector<std::complex<double>> calculateSelfEnergy(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices
	);*/

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

	/** Set U. */
//	void setU(std::complex<double> U);

	/** Set Up. */
//	void setUp(std::complex<double> Up);

	/** Set J. */
//	void setJ(std::complex<double> J);

	/** Set Jp. */
//	void setJp(std::complex<double> Jp);
private:
	/** Momentum space context. */
	const MomentumSpaceContext &momentumSpaceContext;

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

	/** Energies to calculate the self-energy for. */
//	std::vector<std::complex<double>> selfEnergyEnergies;

	/** Flag indicating whether the SelfEnergyCalculator is initialized. */
	bool isInitialized;

	/** Self-energy main loop. */
/*	template<bool singleSelfEnergyEnergy>
	void selfEnergyMainLoop(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices,
		std::vector<std::complex<double>> &result
	);*/

	/** Self-energy main loop. */
	template<bool singleSelfEnergyEnergy>
	void selfEnergyMainLoop(
		const Index &index,
		const std::vector<std::complex<double>> &energies,
		std::vector<std::complex<double>> &result
	);

	/** Interaction parameters. */
//	std::complex<double> U, Up, J, Jp;
};

inline const MomentumSpaceContext& SelfEnergy::getMomentumSpaceContext(
) const{
	return momentumSpaceContext;
}

inline const Property::InteractionVertex& SelfEnergy::getInteractionVertex(
) const{
	return interactionVertex;
}

/*inline void SelfEnergy::setSelfEnergyEnergies(
	const std::vector<std::complex<double>> &selfEnergyEnergies
){
	this->selfEnergyEnergies = selfEnergyEnergies;
}*/

/*inline void SelfEnergy::setU(std::complex<double> U){
	this->U = U;
}

inline void SelfEnergy::setUp(std::complex<double> Up){
	this->Up = Up;
}

inline void SelfEnergy::setJ(std::complex<double> J){
	this->J = J;
}

inline void SelfEnergy::setJp(std::complex<double> Jp){
	this->Jp = Jp;
}*/

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
