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

/** @package TBTKcalc
 *  @file Polynomial.h
 *  @brief Class for storing polynomial expressions.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_POLYNOMIAL
#define COM_DAFER45_TBTK_POLYNOMIAL

#include "TBTK/TBTKMacros.h"

#include <tuple>
#include <vector>

namespace TBTK{

template<typename FieldType, typename VariableType, typename PowerType>
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

	/** Function operator.
	 *
	 *  @param variables The variables to evaluate the polynomial for.
	 *
	 *  @return The value of the polynomial evaluated at the given
	 *  variables. */
	VariableType operator()(const std::vector<VariableType> &variable);

	/** Get the maximum degree of the polynomial.
	 *
	 *  @return A vector containing the maximum degree for each variable.
	 */
	std::vector<PowerType> getMaxDegree() const;

	void print();
private:
	/** The number of variables. */
	unsigned int numVariables;

	/** The polynomial terms. */
	std::vector<std::tuple<FieldType, std::vector<PowerType>>> terms;
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
VariableType Polynomial<FieldType, VariableType, PowerType>::operator()(
	const std::vector<VariableType> &variables
){
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

	return value;
}

template<typename FieldType, typename VariableType, typename PowerType>
std::vector<PowerType> Polynomial<
	FieldType,
	VariableType,
	PowerType
>::getMaxDegree() const{
	std::vector<PowerType> maxDegree;
	for(unsigned int n = 0; n < numVariables; n++)
		maxDegree.push_back(0.);
	for(unsigned int n = 0; n < terms.size(); n++){
		for(unsigned int c = 0; c < numVariables; c++){
			if(std::get<1>(terms[n])[c] > maxDegree[c])
				maxDegree[c] = std::get<1>(terms[n])[c];
		}
	}

	return maxDegree;
}

template<typename FieldType, typename VariableType, typename PowerType>
void Polynomial<FieldType, VariableType, PowerType>::print(){
	for(unsigned int n = 0; n < terms.size(); n++){
		Streams::out << std::get<0>(terms[n]);
		for(unsigned int c = 0; c < numVariables; c++)
			Streams::out << "\t" << std::get<1>(terms[n])[c];
	}
	Streams::out << "\n";
}

}; //End of namespace TBTK

#endif
