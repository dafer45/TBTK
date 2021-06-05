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

/** @file PropertyExtractor.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/PropertyExtractor/PropertyExtractor.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{
namespace PropertyExtractor{

PropertyExtractor::PropertyExtractor(
){
	this->energyType = EnergyType::Real;

	this->energyResolution = ENERGY_RESOLUTION;
	this->lowerBound = LOWER_BOUND;
	this->upperBound = UPPER_BOUND;
	this->energyInfinitesimal = ENERGY_INFINITESIMAL;

	this->lowerFermionicMatsubaraEnergyIndex
		= LOWER_FERMIONIC_MATSUBARA_ENERGY_INDEX;
	this->upperFermionicMatsubaraEnergyIndex
		= UPPER_FERMIONIC_MATSUBARA_ENERGY_INDEX;
	this->lowerBosonicMatsubaraEnergyIndex
		= LOWER_BOSONIC_MATSUBARA_ENERGY_INDEX;
	this->upperBosonicMatsubaraEnergyIndex
		= UPPER_BOSONIC_MATSUBARA_ENERGY_INDEX;
}

PropertyExtractor::~PropertyExtractor(){
}

void PropertyExtractor::setEnergyWindow(
	double lowerBound,
	double upperBound,
	int energyResolution
){
	this->energyType = EnergyType::Real;

	this->energyResolution = energyResolution;
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
}

void PropertyExtractor::setEnergyWindow(
	int lowerFermionicMatsubaraEnergyIndex,
	int upperFermionicMatsubaraEnergyIndex,
	int lowerBosonicMatsubaraEnergyIndex,
	int upperBosonicMatsubaraEnergyIndex
){
	TBTKAssert(
		abs(lowerFermionicMatsubaraEnergyIndex%2) == 1,
		"PropertyExtractor::PropertyExtractor::setEnergyWindow()",
		"'lowerFermionicMatsubaraEnergyIndex="
		<< lowerFermionicMatsubaraEnergyIndex << "' must be odd.",
		""
	);
	TBTKAssert(
		abs(upperFermionicMatsubaraEnergyIndex%2) == 1,
		"PropertyExtractor::PropertyExtractor::setEnergyWindow()",
		"'upperFermionicMatsubaraEnergyIndex="
		<< upperFermionicMatsubaraEnergyIndex << "' must be odd.",
		""
	);
	TBTKAssert(
		abs(lowerBosonicMatsubaraEnergyIndex%2) == 0,
		"PropertyExtractor::PropertyExtractor::setEnergyWindow()",
		"'lowerBosonicMatsubaraEnergyIndex="
		<< lowerBosonicMatsubaraEnergyIndex << "' must be even.",
		""
	);
	TBTKAssert(
		abs(upperBosonicMatsubaraEnergyIndex%2) == 0,
		"PropertyExtractor::PropertyExtractor::setEnergyWindow()",
		"'upperBosonicMatsubaraEnergyIndex="
		<< upperBosonicMatsubaraEnergyIndex << "' must be even.",
		""
	);
	TBTKAssert(
		lowerFermionicMatsubaraEnergyIndex
			<= upperFermionicMatsubaraEnergyIndex,
		"PropertyExtractor::PropertyExtractor::setEnergyWindow()",
		"'lowerFermionicMatsubaraEnergyIndex="
		<< lowerFermionicMatsubaraEnergyIndex << "' must be less or"
		<< " equal to 'upperFermionicMatsubaraEnergyIndex="
		<< upperFermionicMatsubaraEnergyIndex<< "'",
		""
	);
	TBTKAssert(
		lowerBosonicMatsubaraEnergyIndex
			<= upperBosonicMatsubaraEnergyIndex,
		"PropertyExtractor::PropertyExtractor::setEnergyWindow()",
		"'lowerBosonicMatsubaraEnergyIndex="
		<< lowerBosonicMatsubaraEnergyIndex << "' must be less or"
		<< " equal to 'upperBosonicMatsubaraEnergyIndex="
		<< upperBosonicMatsubaraEnergyIndex<< "'",
		""
	);

	this->energyType = EnergyType::Matsubara;

	this->lowerFermionicMatsubaraEnergyIndex
		= lowerFermionicMatsubaraEnergyIndex;
	this->upperFermionicMatsubaraEnergyIndex
		= upperFermionicMatsubaraEnergyIndex;
	this->lowerBosonicMatsubaraEnergyIndex
		= lowerBosonicMatsubaraEnergyIndex;
	this->upperBosonicMatsubaraEnergyIndex
		= upperBosonicMatsubaraEnergyIndex;
}

void PropertyExtractor::setEnergyInfinitesimal(double energyInfinitesimal){
	this->energyInfinitesimal = energyInfinitesimal;
}

Property::Density PropertyExtractor::calculateDensity(
	Index pattern,
	Index ranges
){
	TBTKExit(
		"PropertyExtractor::calculateDensity()",
		"The chosen property extractor does not support this function call.",
		"See the API for list of supported calls."
	);
}

Property::Density PropertyExtractor::calculateDensity(
	vector<Index> patterns
){
	TBTKExit(
		"PropertyExtractor::calculateDensity()",
		"The chosen property extractor does not support this function call.",
		"See the API for list of supported calls."
	);
}

Property::Magnetization PropertyExtractor::calculateMagnetization(
	Index pattern,
	Index ranges
){
	TBTKExit(
		"PropertyExtractor::calculateMagnetization()",
		"The chosen property extractor does not support this function call.",
		"See the API for list of supported calls."
	);
}

Property::Magnetization PropertyExtractor::calculateMagnetization(
	vector<Index> patterns
){
	TBTKExit(
		"PropertyExtractor::calculateMagnetization()",
		"The chosen property extractor does not support this function call.",
		"See the API for list of supported calls."
	);
}

Property::LDOS PropertyExtractor::calculateLDOS(
	Index pattern,
	Index ranges
){
	TBTKExit(
		"PropertyExtractor::calculateLDOS()",
		"The chosen property extractor does not support this function call.",
		"See the API for list of supported calls."
	);
}

Property::LDOS PropertyExtractor::calculateLDOS(
	vector<Index> patterns
){
	TBTKExit(
		"PropertyExtractor::calculateLDOS()",
		"The chosen property extractor does not support this function call.",
		"See the API for list of supported calls."
	);
}

Property::SpinPolarizedLDOS PropertyExtractor::calculateSpinPolarizedLDOS(
	Index pattern,
	Index ranges
){
	TBTKExit(
		"PropertyExtractor::calculateSpinPolarizedLDOS()",
		"The chosen property extractor does not support this function call.",
		"See the API for list of supported calls."
	);
}

Property::SpinPolarizedLDOS PropertyExtractor::calculateSpinPolarizedLDOS(
	vector<Index> patterns
){
	TBTKExit(
		"PropertyExtractor::calculateSpinPolarizedLDOS()",
		"The chosen property extractor does not support this function call.",
		"See the API for list of supported calls."
	);
}

complex<double> PropertyExtractor::calculateExpectationValue(
	Index to,
	Index from
){
	TBTKExit(
		"PropertyExtractor::calculateExpectationValue()",
		"The chosen property extractor does not support this function call.",
		"See the API for list of supported calls."
	);
}

Property::DOS PropertyExtractor::calculateDOS(){
	TBTKExit(
		"PropertyExtractor::calculateDOS()",
		"The chosen property extractor does not support this function call.",
		"See the API for list of supported calls."
	);
}

Property::DOS PropertyExtractor::sampleDOS(
	unsigned int numSamples,
	const vector<Index> &patterns,
	unsigned int seed
){
	bool selectiveSample = true;
	if(patterns.size() == 0)
		selectiveSample = false;

	const Model &model = getSolver().getModel();
	const HoppingAmplitudeSet &hoppingAmplitudeSet
		= model.getHoppingAmplitudeSet();
	unsigned int sampleSpaceSize;
	if(selectiveSample){
		sampleSpaceSize
			= hoppingAmplitudeSet.getIndexListMultiplePatterns(
				patterns
			).size();
	}
	else{
		sampleSpaceSize = model.getBasisSize();
	}
	TBTKAssert(
		sampleSpaceSize > 0,
		"PropertyExtractor::sampleDOS()",
		"The provided 'patterns' does not match any Index in the"
		<< " Model.",
		""
	);

	Property::DOS dos(lowerBound, upperBound, energyResolution);
	srand(seed);
	for(int n = 0; n < (int)numSamples; n++){
		int state = rand()%model.getBasisSize();
		Index index = hoppingAmplitudeSet.getPhysicalIndex(state);
		if(selectiveSample){
			bool acceptedIndex = false;
			for(auto pattern : patterns){
				if(pattern.equals(index, true)){
					acceptedIndex = true;
					break;
				}
			}
			if(!acceptedIndex){
				n--;
				continue;
			}
		}

		Property::LDOS ldos = calculateLDOS({index});
		for(unsigned int n = 0; n < dos.getResolution(); n++)
			dos(n) += ldos(index, n);
	}
	for(unsigned int n = 0; n < dos.getResolution(); n++)
		dos(n) *= sampleSpaceSize/(double)numSamples;

	return dos;
}

double PropertyExtractor::calculateEntropy(){
	TBTKExit(
		"PropertyExtractor::calculateEntropy()",
		"The chosen property extractor does not support this function call.",
		"See the API for list of supported calls."
	);
}

void PropertyExtractor::ensureCompliantRanges(
	const Index &pattern,
	Index &ranges
){
	for(unsigned int n = 0; n < pattern.getSize(); n++){
		if(pattern.at(n) >= 0)
			ranges.at(n) = 1;
	}
}

vector<int> PropertyExtractor::getLoopRanges(
	const Index &pattern,
	const Index &ranges
){
	vector<int> loopRanges;
	for(unsigned int n = 0; n < ranges.getSize(); n++){
		if(pattern.at(n).isRangeIndex())
			loopRanges.push_back(ranges.at(n));
	}

	return loopRanges;
}

IndexTree PropertyExtractor::generateIndexTree(
	std::vector<Index> patterns,
	const HoppingAmplitudeSet &hoppingAmplitudeSet,
	bool keepSummationWildcards,
	bool keepSpinWildcards
){
	IndexTree indexTree;

	for(unsigned int n = 0; n < patterns.size(); n++){
		Index pattern = *(patterns.begin() + n);

		vector<Index> components = pattern.split();

		switch(components.size()){
		case 1:
		{
			pattern = components[0];

			for(unsigned int c = 0; c < pattern.getSize(); c++){
				switch(pattern.at(c)){
				case IDX_ALL:
				case IDX_SUM_ALL:
				case IDX_SPIN:
					pattern.at(c) = IDX_ALL;
					break;
				default:
					TBTKAssert(
						pattern.at(c) >= 0,
						"PropertyExtractor::generateIndexTree()",
						"Subindex " << c << " of pattern " << n << " is invalid.",
						"Must be non-negative, IDX_ALL, IDX_SUM_ALL, or IDX_SPIN."
					);
					break;
				}
				if(pattern.at(c) < 0)
					pattern.at(c) = IDX_ALL;
			}

			vector<Index> indices = hoppingAmplitudeSet.getIndexList(
				pattern
			);
			Index p = *(patterns.begin() + n);
			for(unsigned int c = 0; c < indices.size(); c++){
				for(unsigned int m = 0; m < p.getSize(); m++){
					if(
						keepSummationWildcards
						&& p.at(m).isSummationIndex()
					){
						indices.at(c).at(m) = IDX_SUM_ALL;
					}
					if(
						keepSpinWildcards
						&& p.at(m).isSpinIndex()
					){
						indices.at(c).at(m) = IDX_SPIN;
					}
				}
			}
			for(unsigned int c = 0; c < indices.size(); c++)
				indexTree.add(indices.at(c));

			break;
		}
		case 2:
		{
			IndexTree firstIndexTree = generateIndexTree(
				{components[0]},
				hoppingAmplitudeSet,
				keepSummationWildcards,
				keepSpinWildcards
			);
			IndexTree secondIndexTree = generateIndexTree(
				{components[1]},
				hoppingAmplitudeSet,
				keepSummationWildcards,
				keepSpinWildcards
			);
			for(
				IndexTree::ConstIterator iterator0
					= firstIndexTree.cbegin();
				iterator0 != firstIndexTree.cend();
				++iterator0
			){
				for(
					IndexTree::ConstIterator iterator1
						= secondIndexTree.cbegin();
					iterator1 != secondIndexTree.cend();
					++iterator1
				){
					indexTree.add({*iterator0, *iterator1});
				}
			}

			break;
		}
		default:
			TBTKExit(
				"PropertyExtractor::generateIndexTree()",
				"Only patterns with one and two component"
				<< " Indices are supported so far, but the"
				<< " pattern '" << pattern.toString() << "'"
				<< " has '" << components.size() << "'"
				<< " components.",
				""
			);
		}
	}

	indexTree.generateLinearMap();

	return indexTree;
}

PropertyExtractor::Information::Information(){
	spinIndex = -1;
}

};	//End of namesapce PropertyExtractor
};	//End of namespace TBTK
