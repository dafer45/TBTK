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
 *  @file SelfEnergyCalculator.h
 *  @brief Calculates the self-energy using the RPA approximation.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SOLVER_ELECTRON_FLUCTUATION_VERTEX
#define COM_DAFER45_TBTK_SOLVER_ELECTRON_FLUCTUATION_VERTEX

#include "TBTK/BrillouinZone.h"
#include "TBTK/IndexedDataTree.h"
#include "TBTK/Property/Susceptibility.h"
#include "TBTK/RPA/MomentumSpaceContext.h"
//#include "TBTK/RPA/RPASusceptibilityCalculator.h"

namespace TBTK{
namespace Solver{

class ElectronFluctuationVertex : public Solver, public Communicator{
public:
	/** Constructor. */
	ElectronFluctuationVertex(
		const MomentumSpaceContext &momentumSpaceContext,
		const Property::Susceptibility &chargeSusceptibility,
		const Property::Susceptibility &spinSusceptibility
	);

	/** Get momentum cpsace context. */
	const MomentumSpaceContext& getMomentumSpaceContext() const;

	/** Get the charge susceptibility. */
	const Property::Susceptibility& getChargeSusceptibility() const;

	/** Get the spin susceptibility. */
	const Property::Susceptibility& getSpinSusceptibility() const;

	/** Calculate self-energy vertex. */
	std::vector<std::complex<double>> calculateSelfEnergyVertex(
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

	/** Generate interaction amplitudes. Can be called multiple times and
	 *  will only regenerate the interaction amplitudes when needed. */
	void generateInteractionAmplitudes();
private:
	/** Momentum space context. */
	const MomentumSpaceContext &momentumSpaceContext;

	/** Charge susceptibility. */
	const Property::Susceptibility &chargeSusceptibility;

	/** Spin susceptibility. */
	const Property::Susceptibility &spinSusceptibility;

	/** Flag indicating whether the SelfEnergyCalculator is initialized. */
	bool isInitialized;

	/** Interaction parameters. */
	std::complex<double> U, Up, J, Jp;

	/** Interaction amplitudes used to calculate the self-energy vertex. */
	std::vector<InteractionAmplitude> u1;
	std::vector<InteractionAmplitude> u2;
	std::vector<InteractionAmplitude> u3;

	/** Flag indicating whether the interaction amplitudes are initialized.
	 */
	bool interactionAmplitudesAreGenerated;
};

inline const MomentumSpaceContext&
ElectronFluctuationVertex::getMomentumSpaceContext() const{
	return momentumSpaceContext;
}

inline const Property::Susceptibility&
ElectronFluctuationVertex::getChargeSusceptibility() const{
	return chargeSusceptibility;
}

inline const Property::Susceptibility&
ElectronFluctuationVertex::getSpinSusceptibility() const{
	return spinSusceptibility;
}

inline void ElectronFluctuationVertex::setU(std::complex<double> U){
	this->U = U;
	interactionAmplitudesAreGenerated = false;
}

inline void ElectronFluctuationVertex::setUp(std::complex<double> Up){
	this->Up = Up;
	interactionAmplitudesAreGenerated = false;
}

inline void ElectronFluctuationVertex::setJ(std::complex<double> J){
	this->J = J;
	interactionAmplitudesAreGenerated = false;
}

inline void ElectronFluctuationVertex::setJp(std::complex<double> Jp){
	this->Jp = Jp;
	interactionAmplitudesAreGenerated = false;
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
