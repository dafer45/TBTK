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
		<< lowerBosonicMatsubaraEnergyIndex << "' must be odd.",
		""
	);
	TBTKAssert(
		abs(upperBosonicMatsubaraEnergyIndex%2) == 0,
		"PropertyExtractor::PropertyExtractor::setEnergyWindow()",
		"'upperBosonicMatsubaraEnergyIndex="
		<< upperBosonicMatsubaraEnergyIndex << "' must be odd.",
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
//	initializer_list<Index> patterns
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
//	initializer_list<Index> patterns
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
//	initializer_list<Index>	pattern
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
//	initializer_list<Index> pattern
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

double PropertyExtractor::calculateEntropy(){
	TBTKExit(
		"PropertyExtractor::calculateEntropy()",
		"The chosen property extractor does not support this function call.",
		"See the API for list of supported calls."
	);
}

/*void PropertyExtractor::calculate(
	void (*callback)(
		PropertyExtractor *cb_this,
		void *memory,
		const Index &index,
		int offset
	),
	void *memory,
	Index pattern,
	const Index &ranges,
	int currentOffset,
	int offsetMultiplier
){
	//Find the next specifier index.
	int currentSubindex = pattern.getSize()-1;
	for(; currentSubindex >= 0; currentSubindex--){
		if(pattern.at(currentSubindex) < 0)
			break;
	}

	if(currentSubindex == -1){
		//No further specifier index found. Call the callback.
		callback(this, memory, pattern, currentOffset);
	}
	else{
		//Ensure that the specifier is valid for the Ranges format.
		TBTKAssert(
			pattern.at(currentSubindex) == IDX_SUM_ALL
			|| pattern.at(currentSubindex) == IDX_X
			|| pattern.at(currentSubindex) == IDX_Y
			|| pattern.at(currentSubindex) == IDX_Z,
			"PropertyExtractor::calculate()",
			"Invalid specifier found at subindex "
			<< currentSubindex << ".",
			"Did you mean IDX_SUM_ALL, IDX_X, IDX_Y, or IDX_Z?"
		);

		//Multiply the memory offset for non summation indices.
		int nextOffsetMultiplier = offsetMultiplier;
		if(pattern.at(currentSubindex) != IDX_SUM_ALL)
			nextOffsetMultiplier *= ranges.at(currentSubindex);

		//Set flag indicating whether the current subindex is a
		//summation index.
		bool isSumIndex = false;
		if(pattern.at(currentSubindex) == IDX_SUM_ALL)
			isSumIndex = true;

		//Recurively call the calculate function with the specifier at
		//the current subindex replaced by each subindex value in the
		//corresponding range.
		for(int n = 0; n < ranges.at(currentSubindex); n++){
			pattern.at(currentSubindex) = n;
			calculate(
				callback,
				memory,
				pattern,
				ranges,
				currentOffset,
				nextOffsetMultiplier
			);
			if(!isSumIndex)
				currentOffset += offsetMultiplier;
		}
	}
}*/

void PropertyExtractor::ensureCompliantRanges(
	const Index &pattern,
	Index &ranges
){
	for(unsigned int n = 0; n < pattern.getSize(); n++){
		if(pattern.at(n) >= 0)
			ranges.at(n) = 1;
	}
}

void PropertyExtractor::getLoopRanges(
	const Index &pattern,
	const Index &ranges,
	int *loopDimensions,
	int **loopRanges
){
	*loopDimensions = 0;
	for(unsigned int n = 0; n < ranges.getSize(); n++){
		if(
			pattern.at(n) == IDX_X
			|| pattern.at(n) == IDX_Y
			|| pattern.at(n) == IDX_Z
		){
			(*loopDimensions)++;
		}
	}

	(*loopRanges) = new int[*loopDimensions];
	int counter = 0;
	for(unsigned int n = 0; n < ranges.getSize(); n++){
		if(
			pattern.at(n) == IDX_X
			|| pattern.at(n) == IDX_Y
			|| pattern.at(n) == IDX_Z
		){
			(*loopRanges)[counter++] = ranges.at(n);
		}
	}
}

IndexTree PropertyExtractor::generateIndexTree(
//	std::initializer_list<Index> patterns,
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
						&& p.at(m) == IDX_SUM_ALL
					){
						indices.at(c).at(m) = IDX_SUM_ALL;
					}
					if(keepSpinWildcards && p.at(m) == IDX_SPIN)
						indices.at(c).at(m) = IDX_SPIN;
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

};	//End of namesapce PropertyExtractor
};	//End of namespace TBTK
