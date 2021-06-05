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
 *  @file ExactDiagonalizer.h
 *  @brief Extracts physical properties from the Diagonalizer
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_EXACT_DIAGONALIZER
#define COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_EXACT_DIAGONALIZER

#include "TBTK/Solver/ExactDiagonalizer.h"
#include "TBTK/Property/GreensFunction.h"
#include "TBTK/PropertyExtractor/PropertyExtractor.h"

#include <complex>

namespace TBTK{
namespace PropertyExtractor{

class ExactDiagonalizer : public PropertyExtractor{
public:
	/** Constructor. */
	ExactDiagonalizer(Solver::ExactDiagonalizer &edSolver);

	/** Destructor. */
	~ExactDiagonalizer();

	/** Calculate Green's function. */
	Property::GreensFunction* calculateGreensFunction(
		Index to,
		Index from,
		Property::GreensFunction::Type type = Property::GreensFunction::Type::Retarded
	);

	/** Overrides PropertyExtractor::calculateExpectationValue(). */
	virtual std::complex<double> calculateExpectationValue(
		Index to,
		Index from
	);

	/** Overrides PropertyExtractor::calculateDensity(). */
	virtual Property::Density calculateDensity(
		Index pattern,
		Index ranges
	);

	/** Overrides PropertyExtractor::calculateMagnetization().  */
	virtual Property::Magnetization calculateMagnetization(
		Index pattern,
		Index ranges
	);

	/** Overrides PropertyExtractor::calculateLDOS(). */
	virtual Property::LDOS calculateLDOS(Index pattern, Index ranges);

	/** Overrides PropertyExtractor::calculateSpinPolarizedLDOS(). */
	virtual Property::SpinPolarizedLDOS calculateSpinPolarizedLDOS(
		Index pattern,
		Index ranges
	);
private:
	/** Diagonalizer to work on. */
	Solver::ExactDiagonalizer *edSolver;

	/** Implement PropertyExtractor::getSolver(). */
	virtual const Solver::Solver& getSolver() const;

	/** Callback for calculating density. Used by calculateDensity(). */
	static void calculateDensityCallback(
		PropertyExtractor *cb_this,
		Property::Property &property,
		const Index &index,
		int offset,
		Information &information
	);

	/** Callback for calculating magnetization. Used by
	 *  calculateMangetization(). */
	static void calculateMagnetizationCallback(
		PropertyExtractor *cb_this,
		Property::Property &property,
		const Index &index,
		int offset,
		Information &information
	);

	/** Callback for calculating local density of states. Used by
	 *  calculateLDOS(). */
	static void calculateLDOSCallback(
		PropertyExtractor *cb_this,
		Property::Property &property,
		const Index &index,
		int offset,
		Information &information
	);

	/** Callback for calculating spin-polarized local density of states.
	 *  Used by calculateSpinPolarizedLDOS(). */
	static void calculateSpinPolarizedLDOSCallback(
		PropertyExtractor *cb_this,
		Property::Property &property,
		const Index &index,
		int offset,
		Information &information
	);
};

inline const Solver::Solver& ExactDiagonalizer::getSolver() const{
	return *edSolver;
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK

#endif
/// @endcond
