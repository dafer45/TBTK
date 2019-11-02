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

/** @package TBTKcalc
 *  @file PropertyExtractor.h
 *  @brief Validates patterns that are passed to PropertyExtractor calls.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_PATTERN_VALIDATOR
#define COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_PATTERN_VALIDATOR

#include "TBTK/Index.h"

#include <string>

namespace TBTK{
namespace PropertyExtractor{

/** @brief Validates patterns that are passed to PropertyExtractor calls.
 *
 *  The PatternValidator checks whether a list of @link Index Indices@endlink
 *  satisfies given rules. It is used by PropertyExtractors to verify that the
 *  @link Index Indices@endlink that are passed to its calculate-functions have
 *  the right format.
 *
 *  The PatternValidator is used as follows
 *  ```cpp
 *    PatternValidator patternValidator;
 *    //Configure the PatternValidator here
 *    //...
 *    patternValidator.validate(patterns);
 *  ```
 *  Here *patterns* is a list of Indices that can contain Subindex flags. If
 *  all patterns pass, nothing happens. If some pattern does not pass, an error
 *  message is generated.
 *
 *  # Number of component @link Index Indices@endlink
 *  A multi-component Index is an Index such as {{x, y, z}, {x, y, z}}, which
 *  has two component @link Index Indices@endlink. By default, any number of
 *  component @link Index Indices@endlink are allowed. However, it is possible
 *  to configure the PatternValidator to require a given number of component
 *  @link Index Indices@endlink using
 *  ```cpp
 *    patternValidator.setNumRequiredComponentIndices(2);
 *  ```
 *
 *  # Allowed Subindex flags
 *  By default, no Subindex flags are allowed in the patterns. To allow a
 *  number of Subindex flags to be present in the pattern, add them using, for
 *  example
 *  ```cpp
 *    patternValidator.setAllowedSubindexFlags({IDX_ALL, IDX_SPIN});
 *  ```
 *
 *  # Example
 *  \snippet PropertyExtractor/PatternValidator.cpp PatternValidator
 *  ## Output
 *  \snippet output/PropertyExtractor/PatternValidator.output PatternValidator
 */
class PatternValidator{
public:
	/** Constructor. */
	PatternValidator();

	//TBTKFeature PropertyExtractor.PatternValidator.checkNumRequiredComponentIndices.1 2019-11-01
	//TBTKFeature PropertyExtractor.PatternValidator.checkNumRequiredComponentIndices.2 2019-11-01
	//TBTKFeature PropertyExtractor.PatternValidator.checkNumRequiredComponentIndices.3 2019-11-01
	/** Set the number of required component @link Index Indices@endlink.
	 *  For example, {{x, y, z}, {x, y, z}} has two component @link Index
	 *  Indices@endlink.
	 *
	 *  @param numRequiredComponentIndices The number of component @link
	 *  Index Indices@endlink the pattern must have. The default number -1
	 *  means that any number of components is allowed. */
	void setNumRequiredComponentIndices(int numRequiredComponentIndices);

	//TBTKFeature PropertyExtractor.PatternValidator.checkAllowedSubindexFlags.1 2019-11-01
	//TBTKFeature PropertyExtractor.PatternValidator.checkAllowedSubindexFlags.2 2019-11-01
	//TBTKFeature PropertyExtractor.PatternValidator.checkAllowedSubindexFlags.3 2019-11-01
	//TBTKFeature PropertyExtractor.PatternValidator.checkAllowedSubindexFlags.4 2019-11-01
	//TBTKFeature PropertyExtractor.PatternValidator.checkAllowedSubindexFlags.5 2019-11-01
	/** Specify a list of Subindex flags that are allowed.
	 *
	 *  @param allowedSubindexFlags List of allowed Subindex flags. */
	void setAllowedSubindexFlags(
		const std::vector<Subindex> &allowedSubindexFlags
	);

	/** Set the name of the calling function. Will be included in the error
	 *  message if the validation fails.
	 *
	 *  @param callingFunctionName The name of the calling function. */
	void setCallingFunctionName(const std::string &callingFunctionsName);

	//TBTKFeature PropertyExtractor.PatternValidator.checkNumRequiredComponentIndices.1 2019-11-01
	//TBTKFeature PropertyExtractor.PatternValidator.checkNumRequiredComponentIndices.2 2019-11-01
	//TBTKFeature PropertyExtractor.PatternValidator.checkNumRequiredComponentIndices.3 2019-11-01
	//TBTKFeature PropertyExtractor.PatternValidator.checkAllowedSubindexFlags.1 2019-11-01
	//TBTKFeature PropertyExtractor.PatternValidator.checkAllowedSubindexFlags.2 2019-11-01
	//TBTKFeature PropertyExtractor.PatternValidator.checkAllowedSubindexFlags.3 2019-11-01
	//TBTKFeature PropertyExtractor.PatternValidator.checkAllowedSubindexFlags.4 2019-11-01
	//TBTKFeature PropertyExtractor.PatternValidator.checkAllowedSubindexFlags.5 2019-11-01
	/** Validate patterns. */
	void validate(const std::vector<Index> &patterns) const;
private:
	/** Number of required component Indices. -1 means no requirement. */
	int numRequiredComponentIndices;

	/** The allowed Subindex flags. */
	std::vector<Subindex> allowedSubindexFlags;

	/** The name of the calling function to includd in error messages. */
	std::string callingFunctionName;

	/** Check that a given set of patterns has the correct number of
	 *  component Indices per pattern. Prints error message if not. */
	void validateNumRequiredComponentIndices(
		const std::vector<Index> &patterns
	) const;

	/** Check that a given set of patterns only has non-negative subindices
	 *  or one of the allowed subindex specifiers. Prints error message if
	 *  not. */
	void validateAllowedSubindexFlags(
		const std::vector<Index> &patterns
	) const;
};

inline PatternValidator::PatternValidator(){
	numRequiredComponentIndices = -1;
}

inline void PatternValidator::setNumRequiredComponentIndices(
	int numRequiredComponentIndices
){
	this->numRequiredComponentIndices = numRequiredComponentIndices;
}

inline void PatternValidator::setAllowedSubindexFlags(
	const std::vector<Subindex> &allowedSubindexFlags
){
	this->allowedSubindexFlags = allowedSubindexFlags;
}

inline void PatternValidator::setCallingFunctionName(
	const std::string &callingFunctionName
){
	this->callingFunctionName = callingFunctionName;
}

inline void PatternValidator::validate(
	const std::vector<Index> &patterns
) const{
	if(numRequiredComponentIndices != -1)
		validateNumRequiredComponentIndices(patterns);
	validateAllowedSubindexFlags(patterns);
}

inline void PatternValidator::validateNumRequiredComponentIndices(
	const std::vector<Index> &patterns
) const{
	for(unsigned int n = 0; n < patterns.size(); n++){
		TBTKAssert(
			patterns[n].split().size()
				== (unsigned int)numRequiredComponentIndices,
			callingFunctionName,
			"Unexpected number of pattern component Indices. The"
			<< " pattern was expected to have '"
			<< numRequiredComponentIndices << "', but the pattern"
			<< " '" << patterns[n].toString() << "' has '"
			<< patterns[n].split().size() << "' components.",
			""
		);
	}
}

inline void PatternValidator::validateAllowedSubindexFlags(
	const std::vector<Index> &patterns
) const{
	for(unsigned int n = 0; n < patterns.size(); n++){
		std::vector<Index> components = patterns[n].split();
		for(unsigned int m = 0; m < components.size(); m++){
			for(unsigned int c = 0; c < components[m].getSize(); c++){
				int subindex = components[m][c];
				if(subindex < 0){
					bool isValid = false;
					for(
						unsigned int k = 0;
						k < allowedSubindexFlags.size();
						k++
					){
						if(
							subindex
							== allowedSubindexFlags[k]
						){
							isValid = true;
							break;
						}
					}

					if(!isValid){
						TBTKExit(
							callingFunctionName,
							"Invalid subindex at"
							<< " position '" << c
							<< "' in component"
							<< " Index '" << m
							<< "' of the pattern '"
							<< patterns[n].toString()
							<< "'.",
							""
						);
					}
				}
			}
		}
	}
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK

#endif
