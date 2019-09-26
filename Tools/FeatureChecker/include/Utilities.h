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

/** @package TBTKFeatureChecker
 *  @file Feature.h
 *  @brief Contains information about a feature.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_FEATURE_CHECKER_UTILITIES
#define COM_DAFER45_TBTK_FEATURE_CHECKER_UTILITIES

#include <string>
#include <vector>

namespace TBTK{
namespace FeatureChecker{

std::vector<std::string> splitString(const std::string &str, char delimiter){
	std::vector<std::string> components;
	size_t start;
	size_t end = 0;
	while(
		(start = str.find_first_not_of(delimiter, end))
			!= std::string::npos
	){
		end = str.find(delimiter, start);
		components.push_back(str.substr(start, end - start));
	}

	return components;
}

};	//End of namespace FeatureChecker
};	//End of namespace TBTK

#endif
