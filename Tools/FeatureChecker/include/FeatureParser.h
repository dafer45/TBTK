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
 *  @file FeatureParser.h
 *  @brief Parses files for features.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_FEATURE_CHECKER_FEATURE_PARSER
#define COM_DAFER45_TBTK_FEATURE_CHECKER_FEATURE_PARSER

#include "Feature.h"

#include <string>
#include <vector>

#include "TBTK/json.hpp"

namespace TBTK{
namespace FeatureChecker{

/** @brief Contains information about a feature. */
class FeatureParser{
public:
	static std::vector<Feature> parseJSON(const std::string &filename);
	static std::vector<Feature> parseSourceFile(
		const std::string &filename
	);
private:
	static nlohmann::json openJSON(const std::string &filename);
	static std::vector<std::string> extractFeatureLines(
		const std::string &filenames
	);
};

};	//End of namespace FeatureChecker
};	//End of namespace TBTK

#endif
