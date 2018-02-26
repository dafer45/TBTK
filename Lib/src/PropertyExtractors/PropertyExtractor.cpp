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

#include "PropertyExtractor/PropertyExtractor.h"
#include "TBTKMacros.h"

using namespace std;

namespace TBTK{
namespace PropertyExtractor{

PropertyExtractor::PropertyExtractor(
){
	this->energyResolution = ENERGY_RESOLUTION;
	this->lowerBound = LOWER_BOUND;
	this->upperBound = UPPER_BOUND;
}

PropertyExtractor::~PropertyExtractor(){
}

void PropertyExtractor::setEnergyWindow(
	double lowerBound,
	double upperBound,
	int energyResolution
){
	this->energyResolution = energyResolution;
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
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
	initializer_list<Index> patterns
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
	initializer_list<Index> patterns
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
	initializer_list<Index>	pattern
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
	initializer_list<Index> pattern
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

void PropertyExtractor::calculate(
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
	int currentSubindex = pattern.getSize()-1;
	for(; currentSubindex >= 0; currentSubindex--){
		if(pattern.at(currentSubindex) < 0)
			break;
	}

	if(currentSubindex == -1){
		callback(this, memory, pattern, currentOffset);
	}
	else{
		TBTKAssert(
			pattern.at(currentSubindex) != IDX_ALL,
			"PropertyExtractor::calculate()",
			"IDX_ALL found at subindex " << currentSubindex << ".",
			"Did you mean IDX_SUM_ALL, IDX_X, IDX_Y, IDX_Z, or IDX_SPIN?"
		);

		int nextOffsetMultiplier = offsetMultiplier;
		if(pattern.at(currentSubindex) < IDX_SUM_ALL)
			nextOffsetMultiplier *= ranges.at(currentSubindex);
		bool isSumIndex = false;
		if(pattern.at(currentSubindex) == IDX_SUM_ALL)
			isSumIndex = true;
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

void PropertyExtractor::getLoopRanges(
	const Index &pattern,
	const Index &ranges,
	int *lDimensions,
	int **lRanges
){
	*lDimensions = 0;
	for(unsigned int n = 0; n < ranges.getSize(); n++){
		if(pattern.at(n) < IDX_SUM_ALL)
			(*lDimensions)++;
	}

	(*lRanges) = new int[*lDimensions];
	int counter = 0;
	for(unsigned int n = 0; n < ranges.getSize(); n++){
		if(pattern.at(n) < IDX_SUM_ALL)
			(*lRanges)[counter++] = ranges.at(n);
	}
}

IndexTree PropertyExtractor::generateIndexTree(
	std::initializer_list<Index> patterns,
	const HoppingAmplitudeSet &hoppingAmplitudeSet,
	bool keepSumationWildcards,
	bool keepSpinWildcards
){
	IndexTree indexTree;

	for(unsigned int n = 0; n < patterns.size(); n++){
		Index pattern = *(patterns.begin() + n);
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
				if(keepSumationWildcards && p.at(m) == IDX_SUM_ALL)
					indices.at(c).at(m) = IDX_SUM_ALL;
				if(keepSpinWildcards && p.at(m) == IDX_SPIN)
					indices.at(c).at(m) = IDX_SPIN;
			}
		}
		for(unsigned int c = 0; c < indices.size(); c++)
			indexTree.add(indices.at(c));
	}

	indexTree.generateLinearMap();

	return indexTree;
}

};	//End of namesapce PropertyExtractor
};	//End of namespace TBTK
