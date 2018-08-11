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
 *  @file ElectronFluctuationVertex.h
 *  @brief Extracts physical properties from the
 *  Solver::ElectronFluctuationVertex.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_ELECTRON_FLUCTUATION_VERTEX
#define COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_ELECTRON_FLUCTUATION_VERTEX

#include "TBTK/Solver/ElectronFluctuationVertex.h"
#include "TBTK/Property/InteractionVertex.h"
#include "TBTK/PropertyExtractor/PropertyExtractor.h"

#include <complex>
//#include <initializer_list>

namespace TBTK{
namespace PropertyExtractor{

/** The PropertyExtractor::ElectronFluctuationVertex extracts the
 *  InteractionVertex from Solver::ElectronFluctuationVertex. */
class ElectronFluctuationVertex : public PropertyExtractor{
public:
	/** Constructs a PropertyExtractor::ELectronFluctuationVertex.
	 *
	 *  @param solver The Solver to use. */
	ElectronFluctuationVertex(Solver::ElectronFluctuationVertex &solver);

	/** Overrides PropertyExtractor::setEnergyWindow(). */
	virtual void setEnergyWindow(
		double lowerBound,
		double upperBound,
		int resolution
	);

	/** Overrides PropertyExtractor::setEnergyWindow(). */
	virtual void setEnergyWindow(
		int lowerFermionicMatsubaraEnergyIndex,
		int upperFermionicMatsubaraEnergyIndex,
		int lowerBosonicMatsubaraEnergyIndex,
		int upperBosonicEnergyIndex
	);

	/** Calculates the Susceptibility. */
	virtual Property::InteractionVertex calculateInteractionVertex(
//		std::initializer_list<Index> patterns
		std::vector<Index> patterns
	);
private:
	/** Calback for callculating susceptibility. */
	static void calculateInteractionVertexCallback(
		PropertyExtractor *cb_this,
		void *interactionVertex,
		const Index &index,
		int offset
	);

	/** Solver::ElectronFluctautionVertex to work on. */
	Solver::ElectronFluctuationVertex *solver;

	/** Energies. */
//	std::vector<std::complex<double>> energies;

	//TODO
	//These variables should be made part if the PropertyExtractor instead
	//once its been tested to work well for this specific case.
/*	enum class EnergyType{Real, Matsubara};
	EnergyType energyType;
	int lowerFermionicMatsubaraEnergyIndex;
	int upperFermionicMatsubaraEnergyIndex;
	int lowerBosonicMatsubaraEnergyIndex;
	int upperBosonicMatsubaraEnergyIndex;*/
};

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK

#endif
