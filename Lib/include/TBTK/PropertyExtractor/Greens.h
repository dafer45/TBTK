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
 *  @file Greens.h
 *  @brief Extracts physical properties from the Solver::Greens.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_GREENS
#define COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_GREENS

#include "TBTK/Solver/Greens.h"
#include "TBTK/Property/Density.h"
#include "TBTK/Property/GreensFunction.h"
#include "TBTK/Property/LDOS.h"
#include "TBTK/Property/Magnetization.h"
#include "TBTK/Property/SpinPolarizedLDOS.h"
#include "TBTK/PropertyExtractor/PropertyExtractor.h"

//#include <initializer_list>
#include <iostream>

namespace TBTK{
namespace PropertyExtractor{

/** Experimental class for extracting properties from a Solver::Greens. */
class Greens : public PropertyExtractor{
public:
	/** Constructor. */
	Greens(Solver::Greens &cSolver);

	/** Destructor. */
	virtual ~Greens();

	/** Overrides PropertyExtractor::setEnergyWindow(). */
	virtual void setEnergyWindow(
		double lowerBound,
		double upperBound,
		int energyResolution
	);

	/** Overrides PropertyExtractor::calculateDensity(). */
	virtual Property::Density calculateDensity(
		Index pattern,
		Index ranges
	);

	/** Overrides PropertyExtractor::calculateDensity(). */
	virtual Property::Density calculateDensity(
		std::vector<Index> patterns
	);

	/** Overrides PropertyExtractor::calculateLDOS(). */
	virtual Property::LDOS calculateLDOS(
		std::vector<Index> patterns
	);
private:
	/** ChebyshevExpander to work on. */
	Solver::Greens *solver;

	/** Implements PropertyExtractor::getSolver(). */
	virtual const Solver::Solver& getSolver() const;

	/** Callback for calculating the density. Used by calculateDensity. */
	static void calculateDensityCallback(
		PropertyExtractor *cb_this,
		Property::Property &property,
		const Index &index,
		int offset,
		Information &information
	);

	/** Callback for calculating the local density of states. Used by
	 *  calculateLDOS. */
	static void calculateLDOSCallback(
		PropertyExtractor *cb_this,
		Property::Property &property,
		const Index &index,
		int offset,
		Information &information
	);
};

inline const Solver::Solver& Greens::getSolver() const{
	return *solver;
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK

#endif
/// @endcond
