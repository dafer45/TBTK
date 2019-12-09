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
 *  @file Polynomial.h
 *  @brief Class for storing polynomial expressions.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PADE_APPROXIMATOR
#define COM_DAFER45_TBTK_PADE_APPROXIMATOR

#include "TBTK/Polynomial.h"
#include "TBTK/TBTKMacros.h"

#include <complex>
#include <vector>

namespace TBTK{

class PadeApproximator{
public:
	/** Set the highest allowed degree for the numerator.
	 *
	 *  @param The highest allowed degree for the numerator. */
	void setNumeratorDegree(unsigned int numeratorDegree);

	/** Set the highest allowed degree for the denominator.
	 *
	 *  @param The highest allowed degree for the denominator. */
	void setDenominatorDegree(unsigned int denominatorDegree);

	/** Calculate the Padé approximation for a sampled function.
	 *
	 *  @param values The values of the function at the sample points.
	 *  @param The arguments at which the function has been sampled.
	 *
	 *  @return A vector containing the numerator and denominator
	 *  Polynomial as the first and second component, respectively. */
	std::vector<
		Polynomial<std::complex<double>, std::complex<double>, int>
	> approximate(
		const std::vector<std::complex<double>> &values,
		const std::vector<std::complex<double>> &arguments
	);
private:
	/** The maximum degree of the numerator. */
	unsigned int numeratorDegree;

	/** The maximum degree of the denominator. */
	unsigned int denominatorDegree;

	/** Execute the least square algorithm. */
	void executeLeastSquare(
		std::complex<double> *matrix,
		std::complex<double> *vector,
		unsigned int numRows,
		unsigned int numColumns
	);
};

inline void PadeApproximator::setNumeratorDegree(unsigned int numeratorDegree){
	this->numeratorDegree = numeratorDegree;
}

inline void PadeApproximator::setDenominatorDegree(
	unsigned int denominatorDegree
){
	this->denominatorDegree = denominatorDegree;
}

}; //End of namespace TBTK

#endif
/// @endcond
