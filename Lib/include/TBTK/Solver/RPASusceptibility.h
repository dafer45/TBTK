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
 *  @file RPASuscesptibility.h
 *  @brief Calculates the RPA susceptibility.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_RPA_SUSCEPTIBILITY_CALCULATOR
#define COM_DAFER45_TBTK_RPA_SUSCEPTIBILITY_CALCULATOR

#include "TBTK/InteractionAmplitude.h"
#include "TBTK/Property/Susceptibility.h"
#include "TBTK/RPA/MomentumSpaceContext.h"
#include "TBTK/SerializableVector.h"
#include "TBTK/Solver/Susceptibility.h"

#include <complex>

namespace TBTK{
namespace Solver{

class RPASusceptibility : public Susceptibility, public Communicator{
public:
	/** Constructor. */
	RPASusceptibility(
		const RPA::MomentumSpaceContext &momentumSpaceContext,
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

	/** Calculate charge RPA Susceptibility. */
	IndexedDataTree<
		std::vector<std::complex<double>>
	> calculateChargeRPASusceptibility(
		const Index &index
	);

	/** Calculate spin RPA Susceptibility. */
	IndexedDataTree<
		std::vector<std::complex<double>>
	> calculateSpinRPASusceptibility(
		const Index &index
	);

	/** Set U. */
	void setU(std::complex<double> U);

	/** Set Up. */
	void setUp(std::complex<double> Up);

	/** Set J. */
	void setJ(std::complex<double> J);

	/** Set Jp. */
	void setJp(std::complex<double> Jp);
private:
	/** The bare susceptibility to calculate the RPA susceptibility from. */
	const Property::Susceptibility &bareSusceptibility;

	/** InteractionAmplitudes. */
	std::vector<InteractionAmplitude> interactionAmplitudes;

	/** Interaction amplitudes for charge susceptibility. */
	std::vector<InteractionAmplitude> interactionAmplitudesCharge;

	/** Interaction amplitudes for charge susceptibility. */
	std::vector<InteractionAmplitude> interactionAmplitudesSpin;

	/** Flag indicating whether the interaction amplitudes for charge- and
	 *  spin-susceptibilities has been initialized. */
	bool interactionAmplitudesAreGenerated;

	/** Invert matix. */
	void invertMatrix(
		std::complex<double> *matrix,
		unsigned int dimensions
	);

	/** Multiply matrices. */
/*	void multiplyMatrices(
		std::complex<double> *matrix1,
		std::complex<double> *matrix2,
		std::complex<double> *result,
		unsigned int dimensions
	);*/

	/** RPA-susceptibility main algorithm. */
	std::vector<
		std::vector<std::vector<std::complex<double>>>
	> rpaSusceptibilityMainAlgorithm(
		const Index &index,
		const std::vector<InteractionAmplitude> &interactionAmpltiudes
	);

	/** Interaction parameters. */
	std::complex<double> U, Up, J, Jp;

	/** Generate chare- and spin-interaction amplitudes. Can be called
	 *  multiple times and will only regenerate the interaction amplitudes
	 *  when needed. */
	void generateInteractionAmplitudes();
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

inline void RPASusceptibility::setU(std::complex<double> U){
	this->U = U;
	interactionAmplitudesAreGenerated = false;
}

inline void RPASusceptibility::setUp(std::complex<double> Up){
	this->Up = Up;
	interactionAmplitudesAreGenerated = false;
}

inline void RPASusceptibility::setJ(std::complex<double> J){
	this->J = J;
	interactionAmplitudesAreGenerated = false;
}

inline void RPASusceptibility::setJp(std::complex<double> Jp){
	this->Jp = Jp;
	interactionAmplitudesAreGenerated = false;
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
