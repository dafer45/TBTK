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

/** @file LindhardSusceptibility.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/PropertyExtractor/LindhardSusceptibility.h"
#include "TBTK/Functions.h"
#include "TBTK/Streams.h"
#include "TBTK/Timer.h"

#include <cmath>

using namespace std;

namespace TBTK{
namespace PropertyExtractor{

LindhardSusceptibility::LindhardSusceptibility(
	Solver::LindhardSusceptibility &solver
){
	this->solver = &solver;
}

Property::Susceptibility LindhardSusceptibility::calculateSusceptibility(
	vector<Index> patterns
){
	//Calculate allIndices.
	IndexTree allIndices;
	for(unsigned int n = 0; n < patterns.size(); n++){
		const Index &pattern = *(patterns.begin() + n);

		vector<Index> indices = pattern.split();
		TBTKAssert(
			indices.size() == 5,
			"PropertyExtractor::LindhardSusceptibility::calculateSusceptibility()",
			"Invalid pattern '" << pattern.toString() << "'.",
			"The Index must be a compound Index with five"
			<< " component Indices, but the number of components"
			<< " are " << indices.size() << "."
		);
		for(unsigned int n = 2; n < indices.size(); n++){
			TBTKAssert(
				indices[n].getSize() == indices[1].getSize(),
				"PropertyExtractor::LindhardSusceptibility::calculateSusceptibility()",
				"Currently the last four Indices has to have"
				" the same number of subindices. The clash"
				<< " appeared between '"
				<< indices[1].toString() << "' and '"
				<< indices[n].toString() << "'",
				"Contact the developer if support for more"
				<< " general Indices is needed."
			);
			//TODO
			//For general Indices to be supported, a more general
			//way for finding all possible kIndices is required.
			//The current method relies on knowing the number of
			//subindices to append after the kIndex.
		}

		Index kIndexPattern = indices[0];

		//TODO
		//This is the restricting assumption.
		Index kIndexPatternExtended = kIndexPattern;
		for(unsigned int n = 0; n < indices[1].getSize(); n++)
			kIndexPatternExtended.pushBack(IDX_ALL);

		IndexTree kIndexTree = generateIndexTree(
			{kIndexPatternExtended},
			solver->getModel().getHoppingAmplitudeSet(),
			false,
			false
		);
		for(
			IndexTree::ConstIterator iteratorK
				= kIndexTree.cbegin();
			iteratorK != kIndexTree.cend();
			++iteratorK
		){
			Index kIndex = *iteratorK;
			kIndex = kIndex.getSubIndex(
				0,
				kIndexPattern.getSize()-1
			);

			Index intraBlockIndices[4];
			for(unsigned int n = 0; n < 4; n++){
				intraBlockIndices[n] = Index(
					kIndex,
					indices[n+1]
				);
			}

			IndexTree intraBlockIndicesTree[4];
			for(unsigned int n = 0; n < 4; n++){
				intraBlockIndicesTree[n] = generateIndexTree(
					{intraBlockIndices[n]},
					solver->getModel(
					).getHoppingAmplitudeSet(),
					false,
					false
				);
			}

			for(
				IndexTree::ConstIterator iterator0
					= intraBlockIndicesTree[0].cbegin();
				iterator0 != intraBlockIndicesTree[0].cend();
				++iterator0
			){
				Index index0 = *iterator0;
				index0 = index0.getSubIndex(
					kIndex.getSize(),
					index0.getSize()-1
				);
				for(
					IndexTree::ConstIterator iterator1
						= intraBlockIndicesTree[1].cbegin();
					iterator1
						!= intraBlockIndicesTree[1].cend();
					++iterator1
				){
					Index index1 = *iterator1;
					index1 = index1.getSubIndex(
						kIndex.getSize(),
						index1.getSize()-1
					);
					for(
						IndexTree::ConstIterator iterator2
							= intraBlockIndicesTree[2].cbegin();
						iterator2
							!= intraBlockIndicesTree[2].cend();
						++iterator2
					){
						Index index2 = *iterator2;
						index2 = index2.getSubIndex(
							kIndex.getSize(),
							index2.getSize()-1
						);
						for(
							IndexTree::ConstIterator iterator3
								= intraBlockIndicesTree[3].cbegin();
							iterator3
								!= intraBlockIndicesTree[3].cend();
							++iterator3
						){
							Index index3 = *iterator3;
							index3 = index3.getSubIndex(
								kIndex.getSize(),
								index3.getSize()-1
							);
							allIndices.add({
								kIndex,
								index0,
								index1,
								index2,
								index3
							});
						}
					}
				}
			}
		}
	}
	allIndices.generateLinearMap();

	//Calculate memoryLayout.
	IndexTree memoryLayout;
	for(unsigned int n = 0; n < patterns.size(); n++){
		const Index &pattern = *(patterns.begin() + n);

		vector<Index> indices = pattern.split();
		TBTKAssert(
			indices.size() == 5,
			"PropertyExtractor::LindhardSusceptibility::calculateSusceptibility()",
			"Invalid pattern '" << pattern.toString() << "'.",
			"The Index must be a compound Index with five"
			<< " component Indices, but the number of components"
			<< " are " << indices.size() << "."
		);
		for(unsigned int n = 2; n < indices.size(); n++){
			TBTKAssert(
				indices[n].getSize() == indices[1].getSize(),
				"PropertyExtractor::LindhardSusceptibility::calculateSusceptibility()",
				"Currently the last four Indices has to have"
				" the same number of subindices. The clash"
				<< " appeared between '"
				<< indices[1].toString() << "' and '"
				<< indices[n].toString() << "'",
				"Contact the developer if support for more"
				<< " general Indices is needed."
			);
			//TODO
			//For general Indices to be supported, a more general
			//way for finding all possible kIndices is required.
			//The current method relies on knowing the number of
			//subindices to append after the kIndex.
		}

		Index kIndexPattern = indices[0];

		//TODO
		//This is the restricting assumption.
		Index kIndexPatternExtended = kIndexPattern;
		for(unsigned int n = 0; n < indices[1].getSize(); n++)
			kIndexPatternExtended.pushBack(IDX_ALL);

		IndexTree kIndexTree = generateIndexTree(
			{kIndexPatternExtended},
			solver->getModel().getHoppingAmplitudeSet(),
			true,
			false
		);
		for(
			IndexTree::ConstIterator iteratorK
				= kIndexTree.cbegin();
			iteratorK != kIndexTree.cend();
			++iteratorK
		){
			//Index that prepends the intraBlockIndices.
			Index kIndex = *iteratorK;
			kIndex = kIndex.getSubIndex(
				0,
				kIndexPattern.getSize()-1
			);

			Index intraBlockIndices[4];
			for(unsigned int n = 0; n < 4; n++){
				intraBlockIndices[n] = Index(
					kIndex,
					indices[n+1]
				);
			}

			IndexTree intraBlockIndicesTree[4];
			for(unsigned int n = 0; n < 4; n++){
				intraBlockIndicesTree[n] = generateIndexTree(
					{intraBlockIndices[n]},
					solver->getModel(
					).getHoppingAmplitudeSet(),
					true,
					false
				);
			}

			for(
				IndexTree::ConstIterator iterator0
					= intraBlockIndicesTree[0].cbegin();
				iterator0 != intraBlockIndicesTree[0].cend();
				++iterator0
			){
				Index index0 = *iterator0;
				index0 = index0.getSubIndex(
					kIndex.getSize(),
					index0.getSize()-1
				);
				for(
					IndexTree::ConstIterator iterator1
						= intraBlockIndicesTree[1].cbegin();
					iterator1
						!= intraBlockIndicesTree[1].cend();
					++iterator1
				){
					Index index1 = *iterator1;
					index1 = index1.getSubIndex(
						kIndex.getSize(),
						index1.getSize()-1
					);
					for(
						IndexTree::ConstIterator iterator2
							= intraBlockIndicesTree[2].cbegin();
						iterator2
							!= intraBlockIndicesTree[2].cend();
						++iterator2
					){
						Index index2 = *iterator2;
						index2 = index2.getSubIndex(
							kIndex.getSize(),
							index2.getSize()-1
						);
						for(
							IndexTree::ConstIterator iterator3
								= intraBlockIndicesTree[3].cbegin();
							iterator3
								!= intraBlockIndicesTree[3].cend();
							++iterator3
						){
							Index index3 = *iterator3;
							index3 = index3.getSubIndex(
								kIndex.getSize(),
								index3.getSize()-1
							);
							memoryLayout.add({
								kIndex,
								index0,
								index1,
								index2,
								index3
							});
						}
					}
				}
			}
		}
	}
	memoryLayout.generateLinearMap();

	switch(getEnergyType()){
	case EnergyType::Real:
	{
		double lowerBound = getLowerBound();
		double upperBound = getUpperBound();
		int energyResolution = getEnergyResolution();

		energies.clear();
		energies.reserve(energyResolution);
		double dE;
		if(energyResolution == 1)
			dE = 0;
		else
			dE = (upperBound - lowerBound)/(energyResolution-1);
		for(int n = 0; n < energyResolution; n++)
			energies.push_back(lowerBound + n*dE);

		Property::Susceptibility susceptibility(
			memoryLayout,
			real(energies[0]),
			real(energies.back()),
			energies.size()
		);

		Information information;
		calculate(
			calculateSusceptibilityCallback,
			allIndices,
			memoryLayout,
			susceptibility,
			information
		);

		return susceptibility;
	}
	case EnergyType::Matsubara:
	{
		int lowerBosonicMatsubaraEnergyIndex
			= getLowerBosonicMatsubaraEnergyIndex();
		int upperBosonicMatsubaraEnergyIndex
			= getUpperBosonicMatsubaraEnergyIndex();

		TBTKAssert(
			lowerBosonicMatsubaraEnergyIndex
			<= upperBosonicMatsubaraEnergyIndex,
			"PropertyExtractor::LindhardSusceptibility::calculateSusceptibility()",
			"'lowerBosonicMatsubaraEnergyIndex="
			<< lowerBosonicMatsubaraEnergyIndex << "' must be less"
			<< " or equal to 'upperBosonicMatsubaraEnergyIndex='"
			<< upperBosonicMatsubaraEnergyIndex << "'.",
			"This should never happen, contact the developer."
		);
		unsigned int numMatsubaraEnergies = (
			upperBosonicMatsubaraEnergyIndex
			- lowerBosonicMatsubaraEnergyIndex
		)/2 + 1;

		double temperature = solver->getModel().getTemperature();
		double kT = UnitHandler::getConstantInNaturalUnits("k_B")*temperature;
		double fundamentalMatsubaraEnergy = M_PI*kT;

		energies.clear();
		energies.reserve(numMatsubaraEnergies);
		for(int n = 0; n < (int)numMatsubaraEnergies; n++){
			energies.push_back(
				(double)(
					lowerBosonicMatsubaraEnergyIndex + 2*n
				)*complex<double>(0, 1)*M_PI*kT
			);
		}

		Property::Susceptibility susceptibility(
			memoryLayout,
			lowerBosonicMatsubaraEnergyIndex,
			upperBosonicMatsubaraEnergyIndex,
			fundamentalMatsubaraEnergy
		);

		Information information;
		calculate(
			calculateSusceptibilityCallback,
			allIndices,
			memoryLayout,
			susceptibility,
			information
		);

		return susceptibility;
	}
	default:
		TBTKExit(
			"PropertyExtractor::LindhardSusceptibility::LindhardSusceptibility()",
			"Uknown EnergyType.",
			"This should never happen, contact the developer."
		);
	}
}

void LindhardSusceptibility::calculateSusceptibilityCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	LindhardSusceptibility *propertyExtractor
		= (LindhardSusceptibility*)cb_this;
	Property::Susceptibility &susceptibility
		= (Property::Susceptibility&)property;
	vector<complex<double>> &data = susceptibility.getDataRW();

	vector<complex<double>> s
		= propertyExtractor->solver->calculateSusceptibility(
			index,
			propertyExtractor->energies
		);

	for(unsigned int e = 0; e < s.size(); e++)
		data[offset + e] += s[e];
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
