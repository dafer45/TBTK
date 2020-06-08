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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file RPASusceptibility.h
 *  @brief Extracts physical properties from the
 *  Solver::RPASusceptibility.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_RPA_SUSCEPTIBILITY
#define COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_RPA_SUSCEPTIBILITY

#include "TBTK/Solver/RPASusceptibility.h"
#include "TBTK/Property/Susceptibility.h"
#include "TBTK/PropertyExtractor/PropertyExtractor.h"

#include <complex>

namespace TBTK{
namespace PropertyExtractor{

/** The PropertyExtractor::RPASusceptibility extracts the Susceptibility from
 *  Solver::RPASusceptibility. */
class RPASusceptibility : public PropertyExtractor{
public:
	/** Constructs a PropertyExtractor::RPASusceptibility.
	 *
	 *  @param solver The Solver to use. */
	RPASusceptibility(Solver::RPASusceptibility &solver);

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
	virtual Property::Susceptibility calculateRPASusceptibility(
		std::vector<Index> patterns
	);

	/** Calculates the Susceptibility. */
/*	virtual Property::Susceptibility calculateChargeSusceptibility(
		std::vector<Index> patterns
	);*/

	/** Calculates the Susceptibility. */
/*	virtual Property::Susceptibility calculateSpinSusceptibility(
		std::vector<Index> patterns
	);*/
private:
	/** Calback for callculating susceptibility. */
	static void calculateRPASusceptibilityCallback(
		PropertyExtractor *cb_this,
		Property::Property &property,
		const Index &index,
		int offset,
		Information &information
	);

	/** Calback for callculating susceptibility. */
/*	static void calculateChargeSusceptibilityCallback(
		PropertyExtractor *cb_this,
		Property::Property &property,
		const Index &index,
		int offset,
		Information &information
	);*/

	/** Calback for callculating susceptibility. */
/*	static void calculateSpinSusceptibilityCallback(
		PropertyExtractor *cb_this,
		Property::Property &property,
		const Index &index,
		int offset,
		Information &information
	);*/

	/** RPA susceptibility tree for storing results between calls to the
	 *  calculateRPASusceptibilityCallback(). */
	IndexedDataTree<
		std::vector<std::complex<double>>
	> rpaSusceptibilityTree;

	/** Get the Solver. */
	Solver::RPASusceptibility& getSolver();

	/** Get the Solver. */
	const Solver::RPASusceptibility& getSolver() const;

	/** Charge susceptibility tree for storing results between calls to the
	 *  calculateChargeSusceptibilityCallback(). */
/*	IndexedDataTree<
		std::vector<std::complex<double>>
	> chargeSusceptibilityTree;*/

	/** Spin susceptibility tree for storing results between calls to the
	 *  calculateSpinSusceptibilityCallback(). */
/*	IndexedDataTree<
		std::vector<std::complex<double>>
	> spinSusceptibilityTree;*/
};

inline Solver::RPASusceptibility& RPASusceptibility::getSolver(){
	return PropertyExtractor::getSolver<Solver::RPASusceptibility>();
}

inline const Solver::RPASusceptibility& RPASusceptibility::getSolver() const{
	return PropertyExtractor::getSolver<Solver::RPASusceptibility>();
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK

#endif
/// @endcond
