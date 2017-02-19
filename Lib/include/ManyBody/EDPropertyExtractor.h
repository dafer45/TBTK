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

/** @package TBTKcalc
 *  @file EDPropertyExtractor.h
 *  @brief Extracts physical properties from the DiagonalizationSolver
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ED_PROPERTY_EXTRACTOR
#define COM_DAFER45_TBTK_ED_PROPERTY_EXTRACTOR

#include "ChebyshevSolver.h"
#include "ExactDiagonalizationSolver.h"
#include "PropertyExtractor.h"

#include <complex>

namespace TBTK{

class EDPropertyExtractor : public PropertyExtractor{
public:
	/** Constructor. */
	EDPropertyExtractor(ExactDiagonalizationSolver *edSolver);

	/** Destructor. */
	~EDPropertyExtractor();

	/** Calculate Green's function. */
	std::complex<double>* calculateGreensFunction(
		Index to,
		Index from,
		ChebyshevSolver::GreensFunctionType type = ChebyshevSolver::GreensFunctionType::Retarded
	);

	/**Overrider PropertyExtractor::calculateExpectationValue(). */
	virtual std::complex<double> calculateExpectationValue(Index to, Index from);
private:
	/** DiagonalizationSolver to work on. */
	ExactDiagonalizationSolver *edSolver;
};

};	//End of namespace TBTK

#endif
