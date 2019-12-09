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
 *  @file RPASuscesptibility.h
 *  @brief Calculates the RPA susceptibility.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_RPA_SUSCEPTIBILITY_CALCULATOR
#define COM_DAFER45_TBTK_RPA_SUSCEPTIBILITY_CALCULATOR

#include "TBTK/InteractionAmplitude.h"
#include "TBTK/Property/Susceptibility.h"
#include "TBTK/MomentumSpaceContext.h"
#include "TBTK/SerializableVector.h"
#include "TBTK/Solver/Solver.h"

#include <complex>

namespace TBTK{
namespace Solver{

class RPASusceptibility : public Solver, public Communicator{
public:
	/** Constructor. */
	RPASusceptibility(
		const MomentumSpaceContext &momentumSpaceContext,
		const Property::Susceptibility &susceptibility
	);

	/** Get the bare susceptibility. */
	const Property::Susceptibility& getBareSusceptibility() const;

	/** Create slave RPASusceptibilityCalcuator. The slave reuses internal
	 *  lookup tables used to speed up the calculations and should not be
	 *  used after the generating master have been destructed. */
	RPASusceptibility* createSlave();

	/** Set interaction amplitudes. */
	void setInteractionAmplitudes(
		const std::vector<InteractionAmplitude> &interactionAmplitudes
	);

	/** Calculate Susceptibility (not supported, but prints error message).
	 */
	virtual std::vector<std::complex<double>> calculateSusceptibility(
		const Index &index,
		const std::vector<std::complex<double>> &energies
	);

	/** Calculate RPA Susceptibility. */
	IndexedDataTree<
		std::vector<std::complex<double>>
	> calculateRPASusceptibility(
		const Index &index
	);
private:
	/** The bare susceptibility to calculate the RPA susceptibility from. */
	const Property::Susceptibility &bareSusceptibility;

	/** InteractionAmplitudes. */
	std::vector<InteractionAmplitude> interactionAmplitudes;

	/** MomentumSpaceContext. */
	const MomentumSpaceContext &momentumSpaceContext;

	/** Invert matix. */
	void invertMatrix(
		std::complex<double> *matrix,
		unsigned int dimensions
	);

	/** RPA-susceptibility main algorithm. */
	std::vector<std::vector<std::vector<
		std::vector<std::vector<std::complex<double>>>
	>>> rpaSusceptibilityMainAlgorithm(
		const Index &index,
		const std::vector<InteractionAmplitude> &interactionAmpltiudes
	);
};

inline const Property::Susceptibility&
RPASusceptibility::getBareSusceptibility() const{
	return bareSusceptibility;
}

inline void RPASusceptibility::setInteractionAmplitudes(
	const std::vector<InteractionAmplitude> &interactionAmplitudes
){
	this->interactionAmplitudes = interactionAmplitudes;
}

inline std::vector<std::complex<double>>
RPASusceptibility::calculateSusceptibility(
	const Index &index,
	const std::vector<std::complex<double>> &energies
){
	TBTKExit(
		"Solver::RPSSusceptibility::calculateSusceptibility()",
		"This function is not supported by this Solver.",
		""
	);
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
/// @endcond
