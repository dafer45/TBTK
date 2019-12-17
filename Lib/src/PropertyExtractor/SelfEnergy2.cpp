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

/** @file SelfEnergy2.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/PropertyExtractor/SelfEnergy2.h"
#include "TBTK/Functions.h"
#include "TBTK/Streams.h"
#include "TBTK/Timer.h"

#include <cmath>

using namespace std;

namespace TBTK{
namespace PropertyExtractor{

SelfEnergy2::SelfEnergy2(
	Solver::SelfEnergy2 &solver
){
	this->solver = &solver;
}

Property::SelfEnergy SelfEnergy2::calculateSelfEnergy(
	vector<Index> patterns
){
	//Flag that will be set to false if a block subindex without the
	//IDX_ALL specifier is encountered.
	SelfEnergyBlockInformation information;
	information.setCalculateSelfEnergyForAllBlocks(true);

	//Check input and set calculateSelfEnergyForAllBlocks.
	for(unsigned int n = 0; n < patterns.size(); n++){
		const Index &pattern = *(patterns.begin() + n);

		vector<Index> components = pattern.split();
		TBTKAssert(
			components.size() == 3,
			"PropertyExtractor::SelfEnergy2::calculateSelfEnergy()",
			"Invalid pattern '" << pattern.toString() << "'.",
			"The Index must be a compound Index with five"
			<< " component Indices, but the number of components"
			<< " are " << components.size() << "."
		);
		for(unsigned int c = 2; c < components.size(); c++){
			TBTKAssert(
				components[c].getSize() == components[1].getSize(),
				"PropertyExtractor::SelfEnergy2::calculateSelfEnergy()",
				"Currently the last four Indices has to have"
				" the same number of subindices. The clash"
				<< " appeared between '"
				<< components[1].toString() << "' and '"
				<< components[c].toString() << "'",
				"Contact the developer if support for more"
				<< " general Indices is needed."
			);
			//TODO
			//For general Indices to be supported, a more general
			//way for finding all possible kIndices is required.
			//The current method relies on knowing the number of
			//subindices to append after the kIndex.
		}

		for(unsigned int c = 0; c < components[0].getSize(); c++){
			if(!components[0][c].isWildcard()){
				information.setCalculateSelfEnergyForAllBlocks(
					false
				);
			}
		}
	}

	//Calculate allIndices.
	IndexTree allIndices;
	for(unsigned int n = 0; n < patterns.size(); n++){
		const Index &pattern = *(patterns.begin() + n);
		vector<Index> components = pattern.split();

		Index kIndexPattern = components[0];

		if(information.getCalculateSelfEnergyForAllBlocks()){
			for(
				unsigned int c = 0;
				c < kIndexPattern.getSize();
				c++
			){
				kIndexPattern[c] = 0;
			}
		}

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
			"PropertyExtractor::SelfEnergy2::calculateSelfEnergy()",
			"Invalid pattern '" << pattern.toString() << "'.",
			"The Index must be a compound Index with 3 component"
			<< " Indices, but the number of components are "
			<< components.size() << "."
		);
		for(unsigned int n = 2; n < components.size(); n++){
			TBTKAssert(
				components[n].getSize() == components[1].getSize(),
				"PropertyExtractor::SelfEnergy2::calculateSelfEnergy()",
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
			"PropertyExtractor::SelfEnergy2::calculateSelfEnergy()",
			"'lowerFermionicMatsubaraEnergyIndex="
			<< lowerFermionicMatsubaraEnergyIndex << "' must be less"
			<< " or equal to 'upperFermionicMatsubaraEnergyIndex="
			<< lowerFermionicMatsubaraEnergyIndex << "'.",
			"This should never happen, contact the developer."
		);

		double temperature = solver->getModel().getTemperature();
		double kT = UnitHandler::getConstantInNaturalUnits(
			"k_B"
		)*temperature;
		double fundamentalMatsubaraEnergy = M_PI*kT;

		Property::SelfEnergy selfEnergy(
			memoryLayout,
			lowerFermionicMatsubaraEnergyIndex,
			upperFermionicMatsubaraEnergyIndex,
			fundamentalMatsubaraEnergy
		);

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
			"PropertyExtractor::SelfEnergy2::calculateSelfEnergy()",
			"The InteractionVertex has to have the energy type"
			<< " Property::EnergyResolvedProperty::EnergyType::BosonicMatsubara.",
			""
		);
	}
}

void SelfEnergy2::calculateSelfEnergyCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	SelfEnergy2 *propertyExtractor
		= (SelfEnergy2*)cb_this;
	Property::SelfEnergy &selfEnergy = (Property::SelfEnergy&)property;
	vector<complex<double>> &data = selfEnergy.getDataRW();

	switch(selfEnergy.getEnergyType()){
	case Property::EnergyResolvedProperty<complex<double>>::EnergyType::FermionicMatsubara:
	{
		if(
			(
				(SelfEnergyBlockInformation&)information
			).getCalculateSelfEnergyForAllBlocks()
		){
			vector<Index> components = index.split();

			Property::SelfEnergy s
				= propertyExtractor->solver->calculateSelfEnergyAllBlocks(
					{components[1], components[2]},
					propertyExtractor->getLowerFermionicMatsubaraEnergyIndex(),
					propertyExtractor->getUpperFermionicMatsubaraEnergyIndex()
				);

			const IndexTree &containedIndices
				= s.getIndexDescriptor().getIndexTree();
			for(
				IndexTree::ConstIterator iterator
					= containedIndices.cbegin();
				iterator != containedIndices.cend();
				++iterator
			){
				for(unsigned int e = 0; e < s.getBlockSize(); e++){
					data[selfEnergy.getOffset(*iterator) + e]
						+= s(*iterator, e);
				}
			}
		}
		else{
			vector<complex<double>> se
				= propertyExtractor->solver->calculateSelfEnergy(
					index,
					selfEnergy.getLowerMatsubaraEnergyIndex(),
					selfEnergy.getUpperMatsubaraEnergyIndex()
				);

			for(unsigned int e = 0; e < se.size(); e++)
				data[offset + e] += se[e];
		}

		break;
	}
	default:
		TBTKExit(
			"PropertyExtractor::SelfEnergy2::calculateSelfEnergy()",
			"Only calculations for Matsubara energies supported yet.",
			""
		);
	}
}

SelfEnergy2::SelfEnergyBlockInformation::SelfEnergyBlockInformation(){
	calculateSelfEnergyForAllBlocks = false;
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
