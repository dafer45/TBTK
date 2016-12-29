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
 *  @brief Extracts physical properties from the ChebyshevSolver
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_C_PROPERTY_EXTRACTOR
#define COM_DAFER45_TBTK_C_PROPERTY_EXTRACTOR

#include "PropertyExtractor.h"
#include "ChebyshevSolver.h"
#include "Density.h"
#include "Magnetization.h"
#include "LDOS.h"
#include "SpinPolarizedLDOS.h"

#include <iostream>

namespace TBTK{

/** Experimental class for extracting properties from a ChebyshevSolver. */
class CPropertyExtractor : public PropertyExtractor{
public:
	/** Constructor. */
	CPropertyExtractor(
		ChebyshevSolver *cSolver,
		int numCoefficients,
		int energyResolution,
		bool useGPUToCalculateCoefficients,
		bool useGPUToGenerateGreensFunctions,
		bool useLookupTable = true,
		double lowerBound = -1.,
		double upperBound = 1.
	);

	/** Destructor. */
	~CPropertyExtractor();

	/** Overrides PropertyExtractor::setEnergyWindow(). */
	void setEnergyWindow(
		double lowerBound,
		double upperBound,
		int energyResolution
	);

	/** Calculate Green's function. */
	std::complex<double>* calculateGreensFunction(
		Index to,
		Index from,
		ChebyshevSolver::GreensFunctionType type = ChebyshevSolver::GreensFunctionType::Retarded
	);

	/** Calculate Green's function for a range of 'to'-indices. */
	std::complex<double>* calculateGreensFunctions(
		std::vector<Index> &to,
		Index from,
		ChebyshevSolver::GreensFunctionType type = ChebyshevSolver::GreensFunctionType::Retarded
	);

	/** Overrides PropertyExtractor::calculateExpectationValue(). */
	std::complex<double> calculateExpectationValue(Index to, Index from);

	/** Overrides PropertyExtractor::calculateDensity(). */
	Property::Density* calculateDensity(Index pattern, Index ranges);

	/** Overrides PropertyExtractor::calculateMagnetization(). */
	Property::Magnetization* calculateMagnetization(
		Index pattern,
		Index ranges
	);

	/** Overrides PropertyExtractor::calculateLDOS(). */
	Property::LDOS* calculateLDOS(Index pattern, Index ranges);

	/** Overrides PropertyExtractor::calculateSpinPolarizedLDOS(). */
	Property::SpinPolarizedLDOS* calculateSpinPolarizedLDOS(
		Index pattern,
		Index ranges
	);
private:
	/** ChebyshevSolver to work on. */
	ChebyshevSolver *cSolver;

	/** Number of Chebyshev coefficients used in the expansion. */
	int numCoefficients;

	/** Energy resolution of the Green's function. */
//	int energyResolution;

	/** Lower bound for the evaluation of the Green's function and derived
	 *  properties. */
//	double lowerBound;

	/** Upper bound for the evaluation of the Green's function and derived
	 *  properties. */
//	double upperBound;

	/** Flag indicating whether a lookup table is used or not. */
	bool useLookupTable;

	/** Flag indicating whether the GPU should be used to calculate
	 *  Chebyshev coefficients. */
	bool useGPUToCalculateCoefficients;

	/** Flag indicating whether the GPU should be used to generate Green's
	 *  functions. */
	bool useGPUToGenerateGreensFunctions;

	/** Loops over range indices and calls the appropriate callback
	 *  function to calculate the correct quantity. */
/*	void calculate(
		void (*callback)(
			CPropertyExtractor *cb_this,
			void *memory,
			const Index &index,
			int offset
		),
		void *memory,
		Index pattern,
		const Index &ranges,
		int currentOffset,
		int offsetMultiplier
	);*/

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

	/** Hint used to pass information between calculate[Property] and
	 * calculate[Property]Callback. */
//	void *hint;

	/** Ensure that range indices are on compliant format. (Set range to
	 *  one for indices with non-negative pattern value.) */
//	void ensureCompliantRanges(const Index &pattern, Index &ranges);

	/** Extract ranges for loop indices. */
/*	void getLoopRanges(
		const Index &pattern,
		const Index &ranges,
		int *lDimensions,
		int **lRanges
	);*/
};

};	//End of namespace TBTK

#endif
