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
 *  @brief Extracts physical properties from the
 *  Solver::SelfEnergy2.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_SELF_ENERGY2
#define COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_SELF_ENERGY2

#include "TBTK/Solver/SelfEnergy2.h"
#include "TBTK/Property/SelfEnergy.h"
#include "TBTK/PropertyExtractor/PropertyExtractor.h"

#include <complex>

namespace TBTK{
namespace PropertyExtractor{

/** The PropertyExtractor::SelfEnergy extracts the SelfEnergy from
 *  Solver::SelfEnergy2. */
class SelfEnergy2 : public PropertyExtractor{
public:
	/** Constructs a PropertyExtractor::SelfEnergy2.
	 *
	 *  @param solver The Solver to use. */
	SelfEnergy2(Solver::SelfEnergy2 &solver);

	/** Calculates the Susceptibility. */
	virtual Property::SelfEnergy calculateSelfEnergy(
		std::vector<Index> patterns
	);
private:
	/** Calback for callculating the self-energy. */
	static void calculateSelfEnergyCallback(
		PropertyExtractor *cb_this,
		Property::Property &property,
		const Index &index,
		int offset,
		Information &information
	);

	/** Solver::SelfEnergy2 to work on. */
	Solver::SelfEnergy2 *solver;
};

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK

#endif
