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

/** @file AnalyticalContinuer.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/PadeApproximator.h"
#include "TBTK/Solver/AnalyticalContinuer.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{
namespace Solver{

AnalyticalContinuer::AnalyticalContinuer() : Communicator(true){
	numeratorDegree = 0;
	denumeratorDegree = 0;
	lowerBound = -1;
	upperBound = 1;
	resolution = 1000;
}

Property::GreensFunction AnalyticalContinuer::convert(
	const Property::GreensFunction &greensFunction
) const{
	TBTKAssert(
		greensFunction.getType()
			== Property::GreensFunction::Type::Matsubara,
		"Solver::AnalyticalContinuer::convert()",
		"Ivalid Green's function type. This function is only"
		<< " applicable for Green's functions with type"
		<< " Property::GreensFunction::Type::Matsubara.",
		""
	);

	switch(greensFunction.getIndexDescriptor().getFormat()){
	case IndexDescriptor::Format::Custom:
	{
		const IndexTree &indexTree
			= greensFunction.getIndexDescriptor().getIndexTree();

		Property::GreensFunction newGreensFunction(
			indexTree,
			Property::GreensFunction::Type::Ordinary,
			lowerBound,
			upperBound,
			resolution
		);

		for(
			IndexTree::ConstIterator iterator = indexTree.cbegin();
			iterator != indexTree.cend();
			++iterator
		){
			vector<complex<double>> matsubaraValues;
			vector<complex<double>> matsubaraEnergies;
			for(
				unsigned int n = 0;
				n < greensFunction.getNumMatsubaraEnergies();
				n++
			){
				matsubaraValues.push_back(
					greensFunction(*iterator, n)
				);
				matsubaraEnergies.push_back(
					greensFunction.getMatsubaraEnergy(n)
					+ getModel().getChemicalPotential()
				);
			}

			PadeApproximator padeApproximator;
			padeApproximator.setNumeratorDegree(numeratorDegree);
			padeApproximator.setDenumeratorDegree(
				denumeratorDegree
			);
			vector<Polynomial<>> padePolynomials
				= padeApproximator.approximate(
					matsubaraValues,
					matsubaraEnergies
				);

			for(
				unsigned int n = 0;
				n < newGreensFunction.getResolution();
				n++
			){
				double energy = newGreensFunction.getEnergy(n);
				newGreensFunction(*iterator, n)
					= padePolynomials[0](
						{energy}
					)/padePolynomials[1]({energy});
			}
		}

		return newGreensFunction;
	}
	default:
		TBTKExit(
			"Solver::AnalyticalContinuer::convert()",
			"Only Green's functions on the custom format are"
			<< " supported yet.",
			""
		);
	}
}

};	//End of namespace Solver
};	//End of namespace TBTK
