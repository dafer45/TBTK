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
 *  @file FLEX.h
 *  @brief Executes the fluctuation exchange (FLEX) loop.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SOLVER_FLEX
#define COM_DAFER45_TBTK_SOLVER_FLEX

#include "TBTK/Property/GreensFunction.h"
#include "TBTK/Property/Susceptibility.h"
#include "TBTK/Property/InteractionVertex.h"
#include "TBTK/Property/SelfEnergy.h"
#include "TBTK/RPA/MomentumSpaceContext.h"
#include "TBTK/Solver/Solver.h"

namespace TBTK{
namespace Solver{

class FLEX : public Solver{
public:
	/** Enum class for specifying the current state of the solver. */
	enum class State {
		NotYetStarted,
		GreensFunctionCalculated,
		BareSusceptibilityCalculated,
		RPASusceptibilitiesCalculated,
		InteractionVertexCalculated,
		SelfEnergyCalculated
	};

	/** Enum class for specifying which norm to use when calculating the
	 *  convergence parameter. */
	enum class Norm{Max, L2};

	/** Constructor. */
	FLEX(const MomentumSpaceContext &momentumSpaceContext);

	/** Get momentum cpsace context. */
	const MomentumSpaceContext& getMomentumSpaceContext() const;

	/** Get the GreensFunction.
	 *
	 *  @return The GreensFunction. */
	const Property::GreensFunction& getGreensFunction() const;

	/** Get the bare Susceptibility.
	 *
	 *  @return The bare Susceptibility. */
	const Property::Susceptibility& getBareSusceptibility() const;

	/** Get the RPA spin Susceptibility.
	 *
	 *  @return The RPA spin Susceptibility. */
	const Property::Susceptibility& getRPASpinSusceptibility() const;

	/** Get the RPA charge Susceptibility.
	 *
	 *  @return The RPA charge Susceptibility. */
	const Property::Susceptibility& getRPAChargeSusceptibility() const;

	/** Get the InteractionVertex.
	 *
	 *  @return The InteractionVertex. */
	const Property::InteractionVertex& getInteractionVertex() const;

	/** Get the SelfEnergy.
	 *
	 *  @return The SelfEnergy. */
	const Property::SelfEnergy& getSelfEnergy() const;

	/** Set the energy window used for the calculation.
	 *
	 *  @param lowerFermionicMatsubaraEnergyIndex The lower Fermionic
	 *  Matsubara energy index used for quantities with Fermionic Matsubara
	 *  energies.
	 *
	 *  @param upperFermionicMatsubaraEnergyIndex The upper Fermionic
	 *  Matsubara energy index used for quantities with Fermionic Matsubara
	 *  energies.
	 *
	 *  @param lowerBosonicMatsubaraEnergyIndex The lower Bosonic Matsubara
	 *  energy index used for quantities with Bosonic Matsubara energies.
	 *
	 *  @param upperBosonicMatsubaraEnergyIndex The upper Bosonic Matsubara
	 *  energy index used for quantities with Bosonic Matsubara energies. */
	void setEnergyWindow(
		int lowerFermionicMatsubaraEnergyIndex,
		int upperFermionicMatsubaraEnergyIndex,
		int lowerBosonicMatsubaraEnergyIndex,
		int upperBosonicMatsubaraEnergyIndex
	);

	/** Set the interaction term U.
	 *
	 *  @param U The value of U. */
	void setU(double U);

	/** Set the interaction term J.
	 *
	 *  @param J The value of J. */
	void setJ(double J);

	/** Set the interaction term Up.
	 *
	 *  @param Up The value of Up. */
	void setUp(double Up);

	/** Set the interaction term Jp.
	 *
	 *  @param Jp The value of Jp. */
	void setJp(double Jp);

	/** Get the current state.
	 *
	 *  @return The current state of the solver. */
	State getState() const;

	/** Set the maximum number of iterations.
	 *
	 *  @param maxIterations The maximum number of iteration that the
	 *  self-consistent loop will run. */
	void setMaxIterations(unsigned int maxIterations);

	/** Set the norm used to calculate the convergence parameter.
	 *
	 *  @param norm The norm to use when calculating the convergence
	 *  parameter. */
	void setNorm(Norm norm);

	/** Set the tolerance that is used to terminate the self-consistent
	 *  loop.
	 *
	 *  @param tolerance The tolerance. */
	void setTolerance(double tolerance);

	/** Set callback to be called at each step in the loop.
	 *
	 *  @param callback The callback. Set to nullptr to disable callbacks.
	 */
	void setCallback(void (*callback)(FLEX &solver));

	/** Execute the FLEX loop. */
	void run();
private:
	/** Momentum space context. */
	const MomentumSpaceContext &momentumSpaceContext;

	/** Bare Green's function. */
	Property::GreensFunction greensFunction0;

	/** Green's function. */
	Property::GreensFunction greensFunction;

	/** Green's function. */
	Property::GreensFunction oldGreensFunction;

	/** Susceptibility. */
	Property::Susceptibility bareSusceptibility;

	/** RPA spin Susceptibility. */
	Property::Susceptibility rpaSpinSusceptibility;

	/** RPA charge Susceptibility. */
	Property::Susceptibility rpaChargeSusceptibility;

	/** Interaction vertex. */
	Property::InteractionVertex interactionVertex;

	/** Interaction vertex. */
	Property::SelfEnergy selfEnergy;

	/** The lower Fermionic Matsubara energy index. */
	int lowerFermionicMatsubaraEnergyIndex;

	/** The upper Fermionic Matsubara energy index. */
	int upperFermionicMatsubaraEnergyIndex;

	/** The lower Bosonic Matsubara energy index. */
	int lowerBosonicMatsubaraEnergyIndex;

	/** The upp Bosonic Matsubara energy index. */
	int upperBosonicMatsubaraEnergyIndex;

	/** The interaction term U. */
	double U;

	/** The interaction term J. */
	double J;

	/** The interaction term Up. */
	double Up;

	/** The interaction term Jp. */
	double Jp;

	/** The current state the solver is in. */
	State state;

	/** Maximum numer of iterations. */
	unsigned int maxIterations;

	/** Norm to use when calculating the convergence parameter. */
	Norm norm;

	/** The tolerance that is used to terminate the self-consistent loop.
	 */
	double tolerance;

	/** Parameter used to indicate the degree to which the results have
	 *  converged. */
	double convergenceParameter;

	/** Callback that is called after each step in the loop. */
	void (*callback)(FLEX &solver);

	/** Calculate the bare Green's function. */
	void calculateBareGreensFunction();

	/** Calculate the bare susceptibility. */
	void calculateBareSusceptibility();

	/** Calculate the bare susceptibility. */
	void calculateRPASusceptibilities();

	/** Calculate the interaction vertex. */
	void calculateInteractionVertex();

	/** Calculate the self energy. */
	void calculateSelfEnergy();

	/** Calculate the Green's function. */
	void calculateGreensFunction();

	/** Convert the self-energy index structure from
	 *  {{block}, {intra block 0}, {intra block 1}} to
	 *  {{block, intra block 0}, {block, intra block 1}} */
	void convertSelfEnergyIndexStructure();

	/** Calculate the convergence parameter as the norm of the difference
	 *  between the previous and current Green's Function divided by the
	 *  norm of the previous Green's function. */
	void calculateConvergenceParameter();

	/** Generate the interaction vertex for the RPA charge susceptibility.
	 */
	std::vector<InteractionAmplitude>
		generateRPAChargeSusceptibilityInteractionAmplitudes();

	/** Generate the interaction vertex for the RPA spin susceptibility. */
	std::vector<InteractionAmplitude>
		generateRPASpinSusceptibilityInteractionAmplitudes();
};

inline const MomentumSpaceContext& FLEX::getMomentumSpaceContext() const{
	return momentumSpaceContext;
}

inline const Property::GreensFunction& FLEX::getGreensFunction() const{
	return greensFunction;
}

inline const Property::Susceptibility& FLEX::getBareSusceptibility() const{
	return bareSusceptibility;
}

inline const Property::Susceptibility& FLEX::getRPASpinSusceptibility() const{
	return rpaSpinSusceptibility;
}

inline const Property::Susceptibility& FLEX::getRPAChargeSusceptibility(
) const{
	return rpaChargeSusceptibility;
}

inline const Property::InteractionVertex& FLEX::getInteractionVertex() const{
	return interactionVertex;
}

inline const Property::SelfEnergy& FLEX::getSelfEnergy() const{
	return selfEnergy;
}

inline void FLEX::setEnergyWindow(
	int lowerFermionicMatsubaraEnergyIndex,
	int upperFermionicMatsubaraEnergyIndex,
	int lowerBosonicMatsubaraEnergyIndex,
	int upperBosonicMatsubaraEnergyIndex
){
	TBTKAssert(
		lowerFermionicMatsubaraEnergyIndex
			<= upperFermionicMatsubaraEnergyIndex,
		"Solver::FLEX::setEnergyWindow()",
		"The 'lowerFermionicMatsubaraEnergyIndex="
		<< lowerFermionicMatsubaraEnergyIndex << "' must be less or"
		<< " equal to the 'upperFermionicMatsubaraEnergyIndex="
		<< upperFermionicMatsubaraEnergyIndex << "'.",
		""
	);
	TBTKAssert(
		lowerBosonicMatsubaraEnergyIndex
			<= upperBosonicMatsubaraEnergyIndex,
		"Solver::FLEX::setEnergyWindow()",
		"The 'lowerBosonicMatsubaraEnergyIndex="
		<< lowerBosonicMatsubaraEnergyIndex << "' must be less or"
		<< " equal to the 'upperBosonicMatsubaraEnergyIndex="
		<< upperBosonicMatsubaraEnergyIndex << "'.",
		""
	);

	this->lowerFermionicMatsubaraEnergyIndex
		= lowerFermionicMatsubaraEnergyIndex;
	this->upperFermionicMatsubaraEnergyIndex
		= upperFermionicMatsubaraEnergyIndex;
	this->lowerBosonicMatsubaraEnergyIndex
		= lowerBosonicMatsubaraEnergyIndex;
	this->upperBosonicMatsubaraEnergyIndex
		= upperBosonicMatsubaraEnergyIndex;
}

inline void FLEX::setU(double U){
	this->U = U;
}

inline void FLEX::setJ(double J){
	this->J = J;
}

inline void FLEX::setUp(double Up){
	this->Up = Up;
}

inline void FLEX::setJp(double Jp){
	this->Jp = Jp;
}

inline FLEX::State FLEX::getState() const{
	return state;
}

inline void FLEX::setMaxIterations(unsigned int maxIterations){
	TBTKAssert(
		maxIterations > 0,
		"Solver::FLEX::setMaxIterations()",
		"'maxIterations=" << maxIterations << "' must be larger than"
		<< " zero.",
		""
	);

	this->maxIterations = maxIterations;
}

inline void FLEX::setNorm(Norm norm){
	this->norm = norm;
}

inline void FLEX::setTolerance(double tolerance){
	this->tolerance = tolerance;
}

inline void FLEX::setCallback(void (*callback)(FLEX &solver)){
	this->callback = callback;
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
