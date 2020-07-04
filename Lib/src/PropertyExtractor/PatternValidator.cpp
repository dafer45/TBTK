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

/** @file PatternValidator.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/PropertyExtractor/PatternValidator.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{
namespace PropertyExtractor{

void PatternValidator::validateWaveFunctionPatterns(
	const vector<Index> &patterns
){
	PatternValidator validator;
	validator.setNumRequiredComponentIndices(1);
	validator.setAllowedSubindexFlags({IDX_ALL, IDX_SUM_ALL});
	validator.setCallingFunctionName(
		"PropertyExtractor::PatternValidator::validateWaveFunctionPattern()"
	);
	validator.validate(patterns);
}

void PatternValidator::validateGreensFunctionPatterns(
	const vector<Index> &patterns
){
	PatternValidator validator;
	validator.setNumRequiredComponentIndices(2);
	validator.setAllowedSubindexFlags({IDX_ALL, IDX_SUM_ALL});
	validator.setCallingFunctionName(
		"PropertyExtractor::PatternValidator::validateGreensFunctionPattern()"
	);
	validator.validate(patterns);
}

void PatternValidator::validateDensityPatterns(
	const vector<Index> &patterns
){
	PatternValidator validator;
	validator.setNumRequiredComponentIndices(1);
	validator.setAllowedSubindexFlags({IDX_ALL, IDX_SUM_ALL});
	validator.setCallingFunctionName(
		"PropertyExtractor::PatternValidator::validateDensityPattern()"
	);
	validator.validate(patterns);
}

void PatternValidator::validateMagnetizationPatterns(
	const vector<Index> &patterns
){
	PatternValidator validator;
	validator.setNumRequiredComponentIndices(1);
	validator.setAllowedSubindexFlags({IDX_ALL, IDX_SUM_ALL});
	validator.setRequiredSubindexFlags({{IDX_SPIN, 1}});
	validator.setCallingFunctionName(
		"PropertyExtractor::PatternValidator::validateMagnetizationPattern()"
	);
	validator.validate(patterns);
}

void PatternValidator::validateLDOSPatterns(const vector<Index> &patterns){
	PatternValidator validator;
	validator.setNumRequiredComponentIndices(1);
	validator.setAllowedSubindexFlags({IDX_ALL, IDX_SUM_ALL});
	validator.setCallingFunctionName(
		"PropertyExtractor::PatternValidator::validateLDOSPattern()"
	);
	validator.validate(patterns);
}

void PatternValidator::validateSpinPolarizedLDOSPatterns(
	const vector<Index> &patterns
){
	PatternValidator validator;
	validator.setNumRequiredComponentIndices(1);
	validator.setAllowedSubindexFlags({IDX_ALL, IDX_SUM_ALL});
	validator.setRequiredSubindexFlags({{IDX_SPIN, 1}});
	validator.setCallingFunctionName(
		"PropertyExtractor::PatternValidator::validateSpinPolarizedLDOSPattern()"
	);
	validator.validate(patterns);
}

};	//End of namesapce PropertyExtractor
};	//End of namespace TBTK
