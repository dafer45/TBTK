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

#include "TBTK/Array.h"
#include "TBTK/PadeApproximator.h"
#include "TBTK/Solver/AnalyticalContinuer.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{
namespace Solver{

AnalyticalContinuer::AnalyticalContinuer() : Communicator(true){
	numeratorDegree = 0;
	denominatorDegree = 0;
	lowerBound = -1;
	upperBound = 1;
	resolution = 1000;
	energyInfinitesimal = ENERGY_INFINITESIMAL;
	energyShift = 0.;
	scaleFactor = 1.;
}

Property::GreensFunction AnalyticalContinuer::convert(
	const Property::GreensFunction &greensFunction,
	Property::GreensFunction::Type newType
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
			vector<complex<double>> matsubaraValues;
			vector<complex<double>> matsubaraEnergies;
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
						"Solver::AnalyticalContinuer::convert()",
						"Invalid 'newType'. Only"
						<< " conversion to the"
						<< " retarded and advanced"
						<< " Green's function is"
						<< " supported yet.",
						""
					);
				}

				matsubaraValues.push_back(
					greensFunction(*iterator, n)
				);
				matsubaraEnergies.push_back(
					(
						greensFunction.getMatsubaraEnergy(n)
						+ getModel().getChemicalPotential()
						- energyShift
					)/scaleFactor
				);
			}

			PadeApproximator padeApproximator;
			padeApproximator.setNumeratorDegree(numeratorDegree);
			padeApproximator.setDenominatorDegree(
				denominatorDegree
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
				complex<double> contourDeformation
					= getContourDeformation(
						energy,
						newType
					);

				newGreensFunction(*iterator, n)
					= padePolynomials[0](
						{
							(
								energy
								+ contourDeformation
								- energyShift
							)/scaleFactor
						}
					)/padePolynomials[1]({
						(
							energy
							+ contourDeformation
							- energyShift
						)/scaleFactor
					});
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

complex<double> AnalyticalContinuer::getContourDeformation(
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
			"Solver::AnalyticalContinuer::convert()",
			"Invalid Green's function type. Only contours for the"
			<< " retarded, and advanced Green's functions are"
			<< " supported yet.",
			""
		);
	}
}

};	//End of namespace Solver
};	//End of namespace TBTK
