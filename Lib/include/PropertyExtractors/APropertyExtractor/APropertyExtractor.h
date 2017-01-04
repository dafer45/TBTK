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
 *  @file APropertyExtractor.h
 *  @brief Extracts physical properties from the ArnoldiSolver
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_A_PROPERTY_EXTRACTOR
#define COM_DAFER45_TBTK_A_PROPERTY_EXTRACTOR

#include "../../Solvers/ArnoldiSolver/ArnoldiSolver.h"
#include "DOS.h"
#include "EigenValues.h"
#include "LDOS.h"
#include "PropertyExtractor.h"
#include "SpinPolarizedLDOS.h"

#include <complex>

namespace TBTK{

/** The APropertyExtractor extracts common physical properties such as DOS,
 *  LDOS, etc. from an ArnoldiSolver. These can then be written to file using
 *  the FileWriter. */
class APropertyExtractor : public PropertyExtractor{
public:
	/** Constructor. */
	APropertyExtractor(ArnoldiSolver *aSolver);

	/** Destructor. */
	~APropertyExtractor();

	/** Get eigenvalues. */
	Property::EigenValues* getEigenValues();

	/** Get eigenvalue. */
	double getEigenValue(int state);

	/** Get amplitude for given eigenvector \f$n\f$ and physical index
	 *  \f$x\f$: \f$\Psi_{n}(x)\f$.
	 *  @param state Eigenstate number \f$n\f$
	 *  @param index Physical index \f$x\f$. */
	const std::complex<double> getAmplitude(int state, const Index &index);

	/** Overrides PropertyExtractor::calculateDOS(). */
	virtual Property::DOS* calculateDOS();

	/** Overrides PropertyExtractor::calculateLDOS(). */
	virtual Property::LDOS* calculateLDOS(
		Index pattern,
		Index ranges
	);

	/** Overrides PropertyExtractor::calculateSpinPolarizedLDOS(). */
	virtual Property::SpinPolarizedLDOS* calculateSpinPolarizedLDOS(
		Index pattern,
		Index ranges
	);
private:
	/** Calback for callculating local density of states. Used by
	 *  calculateLDOS. */
	static void calculateLDOSCallback(
		PropertyExtractor *cb_this,
		void *ldos,
		const Index &index,
		int offset
	);

	/** Callback for calculating spin-polarized local density of states.
	 *  Used by calculateSP_LDOS. */
	static void calculateSpinPolarizedLDOSCallback(
		PropertyExtractor *cb_this,
		void *sp_ldos,
		const Index &index,
		int offset
	);

	/** ArnoldiSolver to work on. */
	ArnoldiSolver *aSolver;
};

inline double APropertyExtractor::getEigenValue(int state){
	return aSolver->getEigenValue(state);
}

inline const std::complex<double> APropertyExtractor::getAmplitude(
	int state,
	const Index &index
){
	return aSolver->getAmplitude(state, index);
}

};	//End of namespace TBTK

#endif
