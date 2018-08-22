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
 *  @file SelfEnergy.h
 *  @brief Extracts physical properties from the
 *  Solver::SelfEnergy.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_SELF_ENERGY
#define COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_SELF_ENERGY

#include "TBTK/Solver/SelfEnergy.h"
#include "TBTK/Property/SelfEnergy.h"
#include "TBTK/PropertyExtractor/PropertyExtractor.h"

#include <complex>
//#include <initializer_list>

namespace TBTK{
namespace PropertyExtractor{

/** The PropertyExtractor::SelfEnergy extracts the SelfEnergy from
 *  Solver::SelfEnergy. */
class SelfEnergy : public PropertyExtractor{
public:
	/** Constructs a PropertyExtractor::SelfEnergy.
	 *
	 *  @param solver The Solver to use. */
	SelfEnergy(Solver::SelfEnergy &solver);

	/** Overrides PropertyExtractor::setEnergyWindow(). */
/*	virtual void setEnergyWindow(
		double lowerBound,
		double upperBound,
		int resolution
	);*/

	//TODO
	//This should be extended to become part of the PropertyExtractor
	//interface once its tested to work well for this specific case.
/*	virtual void setEnergyWindow(
		int lowerFermionicMatsubaraEnergyIndex,
		int upperFermionicMatsubaraEnergyIndex,
		int lowerBosonicMatsubaraEnergyIndex,
		int upperBosonicEnergyIndex
	);*/

	/** Calculates the Susceptibility. */
	virtual Property::SelfEnergy calculateSelfEnergy(
//		std::initializer_list<Index> patterns
		std::vector<Index> patterns
	);
private:
	/** Calback for callculating the self-energy. */
	static void calculateSelfEnergyCallback(
		PropertyExtractor *cb_this,
		Property::Property &property,
//		void *selfEnergy,
		const Index &index,
		int offset
	);

	/** Solver::SelfEnergy to work on. */
	Solver::SelfEnergy *solver;

	/** Energies. */
	std::vector<std::complex<double>> energies;

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
