/* Copyright 2016 Kristofer Björnson
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
 *  @file CPropertyExtractor.h
 *  @brief Extracts physical properties from the ChebyshevExpander.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_C_PROPERTY_EXTRACTOR
#define COM_DAFER45_TBTK_C_PROPERTY_EXTRACTOR

#include "Solver/ChebyshevExpander.h"
#include "Density.h"
#include "GreensFunction.h"
#include "LDOS.h"
#include "Magnetization.h"
#include "PropertyExtractor.h"
#include "SpinPolarizedLDOS.h"

#include <initializer_list>
#include <iostream>

namespace TBTK{

/** Experimental class for extracting properties from a ChebyshevExpander. */
class CPropertyExtractor : public PropertyExtractor{
public:
	/** Constructor. */
	CPropertyExtractor(
		Solver::ChebyshevExpander &cSolver,
		int numCoefficients,
		bool useGPUToCalculateCoefficients,
		bool useGPUToGenerateGreensFunctions,
		bool useLookupTable = true
	);

	/** Destructor. */
	virtual ~CPropertyExtractor();

	/** Overrides PropertyExtractor::setEnergyWindow(). */
	virtual void setEnergyWindow(
		double lowerBound,
		double upperBound,
		int energyResolution
	);

	/** Calculate Green's function. */
//	Property::GreensFunction* calculateGreensFunction(
	Property::GreensFunction calculateGreensFunction(
		Index to,
		Index from,
		Property::GreensFunction::Type type = Property::GreensFunction::Type::Retarded
	);

	/** Calculate Green's function for a range of 'to'-indices. */
	Property::GreensFunction calculateGreensFunction(
		std::initializer_list<std::initializer_list<Index>> patterns,
		Property::GreensFunction::Type type = Property::GreensFunction::Type::Retarded
	);

	/** Calculate Green's function for a range of 'to'-indices. */
//	Property::GreensFunction** calculateGreensFunctions(
	Property::GreensFunction calculateGreensFunctions(
		std::vector<Index> &to,
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

	/** Overrides PropertyExtractor::calculateDensity(). */
	virtual Property::Density calculateDensity(
		std::initializer_list<Index> patterns
	);

	/** Overrides PropertyExtractor::calculateMagnetization(). */
	virtual Property::Magnetization calculateMagnetization(
		Index pattern,
		Index ranges
	);

	/** Overrides PropertyExtractor::calculateMagnetization(). */
	virtual Property::Magnetization calculateMagnetization(
		std::initializer_list<Index> patterns
	);

	/** Overrides PropertyExtractor::calculateLDOS(). */
	virtual Property::LDOS calculateLDOS(Index pattern, Index ranges);

	/** Overrides PropertyExtractor::calculateLDOS(). */
	virtual Property::LDOS calculateLDOS(
		std::initializer_list<Index> pattern
	);

	/** Overrides PropertyExtractor::calculateSpinPolarizedLDOS(). */
	virtual Property::SpinPolarizedLDOS calculateSpinPolarizedLDOS(
		Index pattern,
		Index ranges
	);

	/** Overrides PropertyExtractor::calculateSpinPolarizedLDOS(). */
	virtual Property::SpinPolarizedLDOS calculateSpinPolarizedLDOS(
		std::initializer_list<Index> pattern
	);
private:
	/** ChebyshevExpander to work on. */
	Solver::ChebyshevExpander *cSolver;

	/** Number of Chebyshev coefficients used in the expansion. */
	int numCoefficients;

	/** Flag indicating whether a lookup table is used or not. */
	bool useLookupTable;

	/** Flag indicating whether the GPU should be used to calculate
	 *  Chebyshev coefficients. */
	bool useGPUToCalculateCoefficients;

	/** Flag indicating whether the GPU should be used to generate Green's
	 *  functions. */
	bool useGPUToGenerateGreensFunctions;

	/** !!!Not tested!!! Callback for calculating density.
	 *  Used by calculateDensity. */
	static void calculateDensityCallback(
		PropertyExtractor *cb_this,
		void *density,
		const Index &index,
		int offset
	);

	/** !!!Not tested!!! Callback for calculating magnetization.
	 *  Used by calculateMAG. */
	static void calculateMAGCallback(
		PropertyExtractor *cb_this,
		void *density,
		const Index &index,
		int offset
	);

	/** !!!Not tested!!! Callback for calculating local density of states.
	 *  Used by calculateLDOS. */
	static void calculateLDOSCallback(
		PropertyExtractor *cb_this,
		void *ldos,
		const Index &index,
		int offset
	);

	/** !!!Not tested!!! Callback for calculating spin-polarized local
	 *  density of states. Used by calculateSP_LDOS. */
	static void calculateSP_LDOSCallback(
		PropertyExtractor *cb_this,
		void *sp_ldos,
		const Index &index,
		int offset
	);

	/** Ensure that the lookup table is in a ready state. */
	void ensureLookupTableIsReady();
};

};	//End of namespace TBTK

#endif
