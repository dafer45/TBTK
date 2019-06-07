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

/** @file AnalyticalContinuerContinuousFractions.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Array.h"
#include "TBTK/PadeApproximatorContinuousFractions.h"
#include "TBTK/Solver/AnalyticalContinuerContinuousFractions.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"
#include "TBTK/Timer.h"

using namespace std;

namespace TBTK{
namespace Solver{

AnalyticalContinuerContinuousFractions::AnalyticalContinuerContinuousFractions(
) :
	Communicator(true)
{
	lowerBound = -1;
	upperBound = 1;
	resolution = 1000;
	energyInfinitesimal = ENERGY_INFINITESIMAL;
}

Property::GreensFunction AnalyticalContinuerContinuousFractions::convert(
	const Property::GreensFunction &greensFunction,
	Property::GreensFunction::Type newType
) const{
	//Make this a class or function variable.
	const double precision = 256;

	TBTKAssert(
		greensFunction.getType()
			== Property::GreensFunction::Type::Matsubara,
		"Solver::AnalyticalContinuerContinuousFractions::convert()",
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

		Property::GreensFunction newGreensFunction;
		switch(newType){
		case Property::GreensFunction::Type::Retarded:
		case Property::GreensFunction::Type::Advanced:
			newGreensFunction = Property::GreensFunction(
				indexTree,
				newType,
				lowerBound,
				upperBound,
				resolution
			);

			break;
		default:
			TBTKExit(
				"Solver::AnalyticalContinuer::convert()",
				"Invalid 'newType'. Only conversion to the"
				<< " retarded and advanced Green's function is"
				<< " supported yet.",
				""
			);
		}

		for(
			IndexTree::ConstIterator iterator = indexTree.cbegin();
			iterator != indexTree.cend();
			++iterator
		){
			vector<ArbitraryPrecision::Complex> matsubaraValues;
			vector<ArbitraryPrecision::Complex> matsubaraEnergies;
			for(
				unsigned int n = 0;
				n < greensFunction.getNumMatsubaraEnergies();
				n++
			){
				if(newType == Property::GreensFunction::Type::Retarded){
					if(imag(greensFunction.getMatsubaraEnergy(n)) < 0)
						continue;
				}
				else if(newType == Property::GreensFunction::Type::Advanced){
					if(imag(greensFunction.getMatsubaraEnergy(n)) > 0)
						continue;
				}
				else{
					TBTKExit(
						"Solver::AnalyticalContinuerContinuousFractions::convert()",
						"Invalid 'newType'. Only"
						<< " conversion to the"
						<< " retarded and advanced"
						<< " Green's function is"
						<< " supported yet.",
						""
					);
				}

				matsubaraValues.push_back(
					ArbitraryPrecision::Complex(
						precision,
						greensFunction(*iterator, n)
					)
				);
				matsubaraEnergies.push_back(
					ArbitraryPrecision::Complex(
						precision,
						(
							greensFunction.getMatsubaraEnergy(n)
							+ getModel().getChemicalPotential()
						)
					)
				);
			}

			PadeApproximatorContinuousFractions padeApproximator;
			Polynomial<
				ArbitraryPrecision::Complex,
				ArbitraryPrecision::Complex
			> padePolynomial = padeApproximator.approximate(
				matsubaraValues,
				matsubaraEnergies
			);

			for(
				unsigned int n = 0;
				n < newGreensFunction.getResolution();
				n++
			){
				double energy = newGreensFunction.getEnergy(n);
				complex<double> contourDeformation
					= getContourDeformation(
						energy,
						newType
					);

				newGreensFunction(*iterator, n)
					= padePolynomial(
						{
							ArbitraryPrecision::Complex(
								precision,
								energy
								+ contourDeformation
							)
						}
					).getComplexDouble();
			}
		}

		return newGreensFunction;
	}
	default:
		TBTKExit(
			"Solver::AnalyticalContinuerContinuousFractions::convert()",
			"Only Green's functions on the custom format are"
			<< " supported yet.",
			""
		);
	}
}

complex<double> AnalyticalContinuerContinuousFractions::getContourDeformation(
	double energy,
	Property::GreensFunction::Type type
) const{
	switch(type){
	case Property::GreensFunction::Type::Retarded:
		return complex<double>(0, 1)*energyInfinitesimal;
	case Property::GreensFunction::Type::Advanced:
		return complex<double>(0, -1)*energyInfinitesimal;
	default:
		TBTKExit(
			"Solver::AnalyticalContinuerContinuousFractions::convert()",
			"Invalid Green's function type. Only contours for the"
			<< " retarded, and advanced Green's functions are"
			<< " supported yet.",
			""
		);
	}
}

};	//End of namespace Solver
};	//End of namespace TBTK
