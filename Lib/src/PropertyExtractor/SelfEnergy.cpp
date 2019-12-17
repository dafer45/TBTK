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

/** @file SelfEnergy.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/PropertyExtractor/SelfEnergy.h"
#include "TBTK/Functions.h"
#include "TBTK/Streams.h"
#include "TBTK/Timer.h"

#include <cmath>

using namespace std;

namespace TBTK{
namespace PropertyExtractor{

SelfEnergy::SelfEnergy(
	Solver::SelfEnergy &solver
){
	this->solver = &solver;
}

Property::SelfEnergy SelfEnergy::calculateSelfEnergy(
	vector<Index> patterns
){
	//Calculate allIndices.
	IndexTree allIndices;
	for(unsigned int n = 0; n < patterns.size(); n++){
		const Index &pattern = *(patterns.begin() + n);

		vector<Index> components = pattern.split();
		TBTKAssert(
			components.size() == 3,
			"PropertyExtractor::SelfEnergy::calculateSelfEnergy()",
			"Invalid pattern '" << pattern.toString() << "'.",
			"The Index must be a compound Index with five"
			<< " component Indices, but the number of components"
			<< " are " << components.size() << "."
		);
		for(unsigned int n = 2; n < components.size(); n++){
			TBTKAssert(
				components[n].getSize() == components[1].getSize(),
				"PropertyExtractor::SelfEnergy::calculateSelfEnergy()",
				"Currently the last four Indices has to have"
				" the same number of subindices. The clash"
				<< " appeared between '"
				<< components[1].toString() << "' and '"
				<< components[n].toString() << "'",
				"Contact the developer if support for more"
				<< " general Indices is needed."
			);
			//TODO
			//For general Indices to be supported, a more general
			//way for finding all possible kIndices is required.
			//The current method relies on knowing the number of
			//subindices to append after the kIndex.
		}

		Index kIndexPattern = components[0];

		//TODO
		//This is the restricting assumption.
		Index kIndexPatternExtended = kIndexPattern;
		for(unsigned int n = 0; n < components[1].getSize(); n++)
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

			Index intraBlockIndices[2];
			for(unsigned int n = 0; n < 2; n++){
				intraBlockIndices[n] = Index(
					kIndex,
					components[n+1]
				);
			}

			IndexTree intraBlockIndicesTree[2];
			for(unsigned int n = 0; n < 2; n++){
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
					allIndices.add({
						kIndex,
						index0,
						index1
					});
				}
			}
		}
	}
	allIndices.generateLinearMap();

	//Calculate memoryLayout.
	IndexTree memoryLayout;
	for(unsigned int n = 0; n < patterns.size(); n++){
		const Index &pattern = *(patterns.begin() + n);

		vector<Index> components = pattern.split();
		TBTKAssert(
			components.size() == 3,
			"PropertyExtractor::SelfEnergy::calculateSelfEnergy()",
			"Invalid pattern '" << pattern.toString() << "'.",
			"The Index must be a compound Index with 3 component"
			<< " Indices, but the number of components are "
			<< components.size() << "."
		);
		for(unsigned int n = 2; n < components.size(); n++){
			TBTKAssert(
				components[n].getSize() == components[1].getSize(),
				"PropertyExtractor::SelfEnergy::calculateSelfEnergy()",
				"Currently the last four Indices has to have"
				" the same number of subindices. The clash"
				<< " appeared between '"
				<< components[1].toString() << "' and '"
				<< components[n].toString() << "'",
				"Contact the developer if support for more"
				<< " general Indices is needed."
			);
			//TODO
			//For general Indices to be supported, a more general
			//way for finding all possible kIndices is required.
			//The current method relies on knowing the number of
			//subindices to append after the kIndex.
		}

		Index kIndexPattern = components[0];

		//TODO
		//This is the restricting assumption.
		Index kIndexPatternExtended = kIndexPattern;
		for(unsigned int n = 0; n < components[1].getSize(); n++)
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

			Index intraBlockIndices[2];
			for(unsigned int n = 0; n < 2; n++){
				intraBlockIndices[n] = Index(
					kIndex,
					components[n+1]
				);
			}

			IndexTree intraBlockIndicesTree[2];
			for(unsigned int n = 0; n < 2; n++){
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
					memoryLayout.add({
						kIndex,
						index0,
						index1
					});
				}
			}
		}
	}
	memoryLayout.generateLinearMap();

	const Property::InteractionVertex &interactionVertex
		= solver->getInteractionVertex();
	switch(interactionVertex.getEnergyType()){
	case Property::EnergyResolvedProperty<complex<double>>::EnergyType::BosonicMatsubara:
	{
		int lowerFermionicMatsubaraEnergyIndex
			= getLowerFermionicMatsubaraEnergyIndex();
		int upperFermionicMatsubaraEnergyIndex
			= getUpperFermionicMatsubaraEnergyIndex();

		TBTKAssert(
			lowerFermionicMatsubaraEnergyIndex
			<= upperFermionicMatsubaraEnergyIndex,
			"PropertyExtractor::SelfEnergy::calculateSelfEnergy()",
			"'lowerFermionicMatsubaraEnergyIndex="
			<< lowerFermionicMatsubaraEnergyIndex << "' must be less"
			<< " or equal to 'upperFermionicMatsubaraEnergyIndex="
			<< lowerFermionicMatsubaraEnergyIndex << "'.",
			"This should never happen, contact the developer."
		);
		unsigned int numMatsubaraEnergies = (
			upperFermionicMatsubaraEnergyIndex
			- lowerFermionicMatsubaraEnergyIndex
		)/2 + 1;

		double temperature = solver->getModel().getTemperature();
		double kT = UnitHandler::getConstantInNaturalUnits("k_B")*temperature;
		double fundamentalMatsubaraEnergy = M_PI*kT;

		energies.reserve(numMatsubaraEnergies);
		for(int n = 0; n < (int)numMatsubaraEnergies; n++){
			energies.push_back(
				(double)(
					lowerFermionicMatsubaraEnergyIndex + 2*n
				)*complex<double>(0, 1)*M_PI*kT
			);
		}

		Property::SelfEnergy selfEnergy(
			memoryLayout,
			lowerFermionicMatsubaraEnergyIndex,
			upperFermionicMatsubaraEnergyIndex,
			fundamentalMatsubaraEnergy
		);

		Information information;
		calculate(
			calculateSelfEnergyCallback,
			allIndices,
			memoryLayout,
			selfEnergy,
			information
		);

		return selfEnergy;
	}
	default:
		TBTKExit(
			"PropertyExtractor::ElectronFluctuationVertex::calculateInteractionVertex()",
			"The InteractionVertex has to have the energy type"
			<< " Property::EnergyResolvedProperty::EnergyType::BosonicMatsubara.",
			""
		);
	}
}

void SelfEnergy::calculateSelfEnergyCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	SelfEnergy *propertyExtractor
		= (SelfEnergy*)cb_this;
	Property::SelfEnergy &selfEnergy = (Property::SelfEnergy&)property;
	vector<complex<double>> &data = selfEnergy.getDataRW();

	vector<complex<double>> se
		= propertyExtractor->solver->calculateSelfEnergy(
			index,
			propertyExtractor->energies
		);

	for(unsigned int e = 0; e < se.size(); e++)
		data[offset + e] += se[e];
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
