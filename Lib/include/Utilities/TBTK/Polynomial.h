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

#ifndef COM_DAFER45_TBTK_POLYNOMIAL
#define COM_DAFER45_TBTK_POLYNOMIAL

#include "TBTK/TBTKMacros.h"

#include <complex>
#include <tuple>
#include <vector>

namespace TBTK{

template<
	typename FieldType = std::complex<double>,
	typename VariableType = std::complex<double>,
	typename PowerType = int
>
class Polynomial{
public:
	/** Constructs a Polynomial.
	 *
	 *  @param numVariables The number of variable parameters. */
	Polynomial(unsigned int numVariables);

	/** Add term.
	 *
	 *  @param coefficient The coefficient of the term.
	 *  @param powers The powers for the term. The number of powers must be
	 *  the same as the number of variables specified in the constructor.
	 */
	void addTerm(
		const FieldType &coefficient,
		const std::vector<PowerType> &powers
	);

	/** Add variable power term.
	 *
	 *  @param coefficient The coefficient of the term.
	 *  @param powers The powers for the term. The number of powers must be
	 *  the same as the number of variables specified in the constructor.
	 */
	void addTerm(
		const FieldType &coefficient,
		const std::initializer_list<PowerType> &powers
	){
		addTerm(coefficient, std::vector<PowerType>(powers));
	}

	/** Add polynomial term.
	 *
	 *  @param term Polynomial to add as term.
	 *  @param power The power of the term. */
	void addTerm(const Polynomial &term, PowerType power);

	/** Add polynomial product term.
	 *
	 *  @param lhs The left hand side of the polynomial product.
	 *  @param rhs The right hand side of the polynomial product. */
	void addTerm(const Polynomial &lhs, const Polynomial &rhs);


	/** Function operator.
	 *
	 *  @param variables The variables to evaluate the polynomial for.
	 *
	 *  @return The value of the polynomial evaluated at the given
	 *  variables. */
	VariableType operator()(const std::vector<VariableType> &variable) const;
private:
	/** The number of variables. */
	unsigned int numVariables;

	/** Terms that are powers of the variable. */
	std::vector<std::tuple<FieldType, std::vector<PowerType>>> terms;

	/** Terms that are powers of polynomials of the variable. */
	std::vector<std::tuple<Polynomial, PowerType>> polynomialTerms;

	/** Terms that are polynomial products. */
	std::vector<std::tuple<Polynomial, Polynomial>> productTerms;
};

template<typename FieldType, typename VariableType, typename PowerType>
Polynomial<FieldType, VariableType, PowerType>::Polynomial(
	unsigned int numVariables
){
	this->numVariables = numVariables;
}

template<typename FieldType, typename VariableType, typename PowerType>
void Polynomial<FieldType, VariableType, PowerType>::addTerm(
	const FieldType &coefficient,
	const std::vector<PowerType> &powers
){
	TBTKAssert(
		powers.size() == numVariables,
		"Polynomial::addTerm()",
		"The number of powers '" << powers.size() << "' must be equal"
		<< " to the number of variables '" << numVariables << "' in"
		<< " the polynomial.",
		""
	);

	terms.push_back(std::make_tuple(coefficient, powers));
}

template<typename FieldType, typename VariableType, typename PowerType>
void Polynomial<FieldType, VariableType, PowerType>::addTerm(
	const Polynomial &term,
	PowerType power
){
	TBTKAssert(
		term.numVariables == numVariables,
		"Polynomial::addTerm()",
		"The term has to have the same number of variables as the"
		<< " total polynomial. The term has '" << term.numVariables
		<< "' variables, while the Polynomial has '" << numVariables
		<< "' variables.",
		""
	);

	polynomialTerms.push_back(std::make_tuple(term, power));
}

template<typename FieldType, typename VariableType, typename PowerType>
void Polynomial<FieldType, VariableType, PowerType>::addTerm(
	const Polynomial &lhs,
	const Polynomial &rhs
){
	TBTKAssert(
		lhs.numVariables == numVariables,
		"Polynomial::addTerm()",
		"Incompatible number of variables. The left hand side is a"
		<< " polynomial with '" << lhs.numVariables << "', while the"
		<< " polynomial that the product is added to has '"
		<< numVariables << "' variables. The number of variables must"
		<< " be the same.",
		""
	);
	TBTKAssert(
		rhs.numVariables == numVariables,
		"Polynomial::addTerm()",
		"Incompatible number of variables. The right hand side is a"
		<< " polynomial with '" << rhs.numVariables << "', while the"
		<< " polynomial that the product is added to has '"
		<< numVariables << "' variables. The number of variables must"
		<< " be the same.",
		""
	);

	productTerms.push_back(std::make_tuple(lhs, rhs));
}

template<typename FieldType, typename VariableType, typename PowerType>
VariableType Polynomial<FieldType, VariableType, PowerType>::operator()(
	const std::vector<VariableType> &variables
) const{
	TBTKAssert(
		variables.size() == numVariables,
		"Polynomial::operator()",
		"The number of given variables are '" << variables.size()
		<< "', but the polynomial has '" << numVariables << "'"
		<< " variables.",
		""
	);

	VariableType value = 0.;
	for(unsigned int n = 0; n < terms.size(); n++){
		VariableType term = std::get<0>(terms[n]);
		for(unsigned int c = 0; c < numVariables; c++){
			PowerType power = std::get<1>(terms[n])[c];
			if(power != 0)
				term *= pow(variables[c], power);
		}

		value += term;
	}

	for(unsigned int n = 0; n < polynomialTerms.size(); n++){
		VariableType term = pow(
			std::get<0>(polynomialTerms[n])(variables),
			std::get<1>(polynomialTerms[n])
		);

		value += term;
	}

	for(unsigned int n = 0; n < productTerms.size(); n++){
		const Polynomial &lhs = std::get<0>(productTerms[n]);
		const Polynomial &rhs = std::get<1>(productTerms[n]);
		VariableType term = lhs(variables)*rhs(variables);

		value += term;
	}

	return value;
}

}; //End of namespace TBTK

#endif
/// @endcond
