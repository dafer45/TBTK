/* Copyright 2019 Kristofer Björnson
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

/** @file PadeApproximatorContinuousFractions.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/PadeApproximatorContinuousFractions.h"

using namespace std;

namespace TBTK{

Polynomial<
	ArbitraryPrecision::Complex,
	ArbitraryPrecision::Complex,
	int
> PadeApproximatorContinuousFractions::approximate(
	const vector<ArbitraryPrecision::Complex> &values,
	const vector<ArbitraryPrecision::Complex> &arguments
){
	TBTKAssert(
		values.size() == arguments.size(),
		"PadeApproximatorContinuousFractions::approximate()",
		"Incompatible sizes. The size of 'values' (" << values.size()
		<< ") must be the same as the size of 'arguments' ("
		<< arguments.size() << ").",
		""
	);

	for(unsigned int n = 0; n < values.size(); n++){
		complex<double> value = values[n].getComplexDouble();
		if(real(value) == 0 && imag(value) == 0)
			return Polynomial<
				ArbitraryPrecision::Complex,
				ArbitraryPrecision::Complex,
				int
			>(1);
	}

	ArbitraryPrecision::Complex g[values.size()];
	for(unsigned int n = 0; n < values.size(); n++)
		g[n] = values[n];

	for(unsigned int n = 1; n < values.size(); n++){
		for(unsigned int c = n; c < values.size(); c++){
			g[c] = (g[n-1] - g[c])/((arguments[c] - arguments[n-1])*g[c]);
		}
	}

	mp_bitcnt_t precision = values[0].getReal().getPrecision();

	Polynomial<
		ArbitraryPrecision::Complex,
		ArbitraryPrecision::Complex,
		int
	> polynomial(1);
	polynomial.addTerm(g[values.size()-1], {0});
	for(int n = values.size()-2; n >= 0; n--){
		Polynomial<
			ArbitraryPrecision::Complex,
			ArbitraryPrecision::Complex,
			int
		> denominator0(1);
		denominator0.addTerm(ArbitraryPrecision::Complex(precision, 1, 0), {0});

		Polynomial<
			ArbitraryPrecision::Complex,
			ArbitraryPrecision::Complex,
			int
		> zMinusZ(1);
		zMinusZ.addTerm(ArbitraryPrecision::Complex(precision, 1, 0), {1});
		zMinusZ.addTerm(
			-arguments[n],
			{0}
		);

		Polynomial<
			ArbitraryPrecision::Complex,
			ArbitraryPrecision::Complex,
			int
		> denominator1(1);
		denominator1.addTerm(polynomial, zMinusZ);

		Polynomial<
			ArbitraryPrecision::Complex,
			ArbitraryPrecision::Complex,
			int
		> denominator(1);
		denominator.addTerm(denominator0, 1);
		denominator.addTerm(denominator1, 1);

		Polynomial<
			ArbitraryPrecision::Complex,
			ArbitraryPrecision::Complex,
			int
		> denominatorInverted(1);
		denominatorInverted.addTerm(denominator, -1);

		Polynomial<
			ArbitraryPrecision::Complex,
			ArbitraryPrecision::Complex,
			int
		> numerator(1);
		numerator.addTerm(g[n], {0});

		polynomial = Polynomial<
			ArbitraryPrecision::Complex,
			ArbitraryPrecision::Complex,
			int
		>(1);
		polynomial.addTerm(numerator, denominatorInverted);
	}

	return polynomial;
}

};	//End of namespace TBTK
