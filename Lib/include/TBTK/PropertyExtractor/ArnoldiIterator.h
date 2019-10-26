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
 *  @brief Extracts physical properties from the Solver::ArnoldiIterator.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_ARNOLDI_ITERATOR
#define COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_ARNOLDI_ITERATOR

#include "TBTK/Solver/ArnoldiIterator.h"
#include "TBTK/Property/DOS.h"
#include "TBTK/Property/EigenValues.h"
#include "TBTK/Property/GreensFunction.h"
#include "TBTK/Property/LDOS.h"
#include "TBTK/Property/SpinPolarizedLDOS.h"
#include "TBTK/Property/WaveFunctions.h"
#include "TBTK/PropertyExtractor/PropertyExtractor.h"

#include <complex>
//#include <initializer_list>

namespace TBTK{
namespace PropertyExtractor{

/** @brief Extracts physical properties from the Solver::ArnoldiIterator.
 *
 *  The PropertyExtractor::ArnoldiIterator extracts @link
 *  Property::AbstractProperty Properties@endlink from the
 *  Solver::ArnoldiIterator.
 *
 *  # Example
 *  \snippet PropertyExtractor/ArnoldiIterator.cpp ArnoldiIterator
 *  ## Output
 *  \snippet output/PropertyExtractor/ArnoldiIterator.output ArnoldiIterator */
class ArnoldiIterator : public PropertyExtractor{
public:
	/** Constructs a PropertyExtractor::ArnoldiIterator.
	 *
	 *  @param solver Solver to use. */
	ArnoldiIterator(Solver::ArnoldiIterator &solver);

	/** Get eigenvalues. */
	Property::EigenValues getEigenValues();

	/** Get eigenvalue. */
	double getEigenValue(int state);

	/** Get amplitude for given eigenvector \f$n\f$ and physical index
	 *  \f$x\f$: \f$\Psi_{n}(x)\f$.
	 *  @param state Eigenstate number \f$n\f$
	 *  @param index Physical index \f$x\f$. */
	const std::complex<double> getAmplitude(int state, const Index &index);

	/** Calculate wave function. */
	Property::WaveFunctions calculateWaveFunctions(
		std::vector<Index> patterns,
		std::vector<Subindex> states
	);

	/** Calculate Green's function. */
/*	Property::GreensFunction* calculateGreensFunction(
		Index to,
		Index from,
		Property::GreensFunction::Type type = Property::GreensFunction::Type::Retarded
	);*/

	/** Overrides PropertyExtractor::calculateDOS(). */
	virtual Property::DOS calculateDOS();

	/** Overrides PropertyExtractor::calculateLDOS(). */
	virtual Property::LDOS calculateLDOS(
		Index pattern,
		Index ranges
	);

	/** Overrides PropertyExtractor::calculateLDOS(). */
	virtual Property::LDOS calculateLDOS(
		std::vector<Index> patterns
	);

	/** Overrides PropertyExtractor::calculateSpinPolarizedLDOS(). */
	virtual Property::SpinPolarizedLDOS calculateSpinPolarizedLDOS(
		Index pattern,
		Index ranges
	);

	/** Overrides PropertyExtractor::calculateSpinPolarizedLDOS(). */
	virtual Property::SpinPolarizedLDOS calculateSpinPolarizedLDOS(
		std::vector<Index> patterns
	);
private:
	/** Callback for calculating the wave function. Used by
	 *  calculateWaveFunctions. */
	static void calculateWaveFunctionsCallback(
		PropertyExtractor *cb_this,
		Property::Property &property,
		const Index &index,
		int offset,
		Information &information
	);

	/** Callback for callculating local density of states. Used by
	 *  calculateLDOS. */
	static void calculateLDOSCallback(
		PropertyExtractor *cb_this,
		Property::Property &property,
		const Index &index,
		int offset,
		Information &information
	);

	/** Callback for calculating spin-polarized local density of states.
	 *  Used by calculateSP_LDOS. */
	static void calculateSpinPolarizedLDOSCallback(
		PropertyExtractor *cb_this,
		Property::Property &property,
		const Index &index,
		int offset,
		Information &information
	);

	/** Solver::ArnoldiIterator to work on. */
	Solver::ArnoldiIterator *aSolver;
};

inline double ArnoldiIterator::getEigenValue(int state){
	return aSolver->getEigenValue(state);
}

inline const std::complex<double> ArnoldiIterator::getAmplitude(
	int state,
	const Index &index
){
	return aSolver->getAmplitude(state, index);
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK

#endif
