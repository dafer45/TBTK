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
 *  @file DPropertyExtractor.h
 *  @brief Extracts physical properties from the Diagonalizer
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_DIAGONALIZER
#define COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_DIAGONALIZER

#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Property/DOS.h"
#include "TBTK/Property/Density.h"
#include "TBTK/Property/EigenValues.h"
#include "TBTK/Property/GreensFunction.h"
#include "TBTK/Property/LDOS.h"
#include "TBTK/Property/Magnetization.h"
#include "TBTK/Property/SpinPolarizedLDOS.h"
#include "TBTK/Property/WaveFunctions.h"
#include "TBTK/PropertyExtractor/PropertyExtractor.h"

#include <complex>
#include <initializer_list>

namespace TBTK{
namespace PropertyExtractor{

/** The PropertyExtractor::Diagonalizer extracts common physical properties
 *  such as DOS, Density, LDOS, etc. from a Diagonalizer. These can then be
 *  written to file using the FileWriter.*/
class Diagonalizer : public PropertyExtractor{
public:
	/** Constructor. */
	Diagonalizer(Solver::Diagonalizer &dSolver);

	/** Destructor. */
	virtual ~Diagonalizer();

	/** Legacy. */
	void saveEigenValues(
		std::string path = "./",
		std::string filename = "EV.dat"
	);

	/** Experimental. Extracts a tabulated version of the
	 *  HoppingAmplitudeSet. */
	void getTabulatedHoppingAmplitudeSet(
		std::complex<double> **amplitudes,
		int **indices,
		int *numHoppingAmplitudes,
		int *maxIndexSize
	);

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
		std::initializer_list<Index> patterns,
		std::initializer_list<int> states
	);

	/** Calculate Green's function. */
/*	Property::GreensFunction* calculateGreensFunction(
		Index to,
		Index from,
		Property::GreensFunction::Type type = Property::GreensFunction::Type::Retarded
	);*/

	/** Overrides PropertyExtractor::calculateDOS(). */
	virtual Property::DOS calculateDOS();

	/** Calculate expectation value. */
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
	virtual Property::LDOS calculateLDOS(
		Index pattern,
		Index ranges
	);

	/** Overrides PropertyExtractor::calculateLDOS(). */
	virtual Property::LDOS calculateLDOS(
		std::initializer_list<Index> patterns
	);

	/** Overrides PropertyExtractor::calculateSpinPolarizedLDOS(). */
	virtual Property::SpinPolarizedLDOS calculateSpinPolarizedLDOS(
		Index pattern,
		Index ranges
	);

	/** Overrides PropertyExtractor::calculateSpinPolarizedLDOS(). */
	virtual Property::SpinPolarizedLDOS calculateSpinPolarizedLDOS(
		std::initializer_list<Index> patterns
	);

	/** Overrides PropertyExtractor::calculateEntropy(). */
	virtual double calculateEntropy();
private:
	/** Callback for calculating the wave function. Used by
	 *  calculateWaveFunctions. */
	static void calculateWaveFunctionsCallback(
		PropertyExtractor *cb_this,
		void *waveFunctions,
		const Index &index,
		int offset
	);

	/** Callback for calculating density. Used by calculateDensity. */
	static void calculateDensityCallback(
		PropertyExtractor *cb_this,
		void *density,
		const Index &index,
		int offset
	);

	/** Callback for calculating magnetization. Used by calculateMAG. */
	static void calculateMAGCallback(
		PropertyExtractor *cb_this,
		void *mag,
		const Index &index,
		int offset
	);

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
	static void calculateSP_LDOSCallback(
		PropertyExtractor *cb_this,
		void *sp_ldos,
		const Index &index,
		int offset
	);

	/** Solver::Diagonalizer to work on. */
	Solver::Diagonalizer *dSolver;
};

inline double Diagonalizer::getEigenValue(int state){
	return dSolver->getEigenValue(state);
}

inline const std::complex<double> Diagonalizer::getAmplitude(
	int state,
	const Index &index
){
	return dSolver->getAmplitude(state, index);
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK

#endif
