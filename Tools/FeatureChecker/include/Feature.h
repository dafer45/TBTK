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

#ifndef COM_DAFER45_TBTK_FEATURE_CHECKER_FEATURE
#define COM_DAFER45_TBTK_FEATURE_CHECKER_FEATURE

#include <string>

namespace TBTK{
namespace FeatureChecker{

/** @brief Contains information about a feature. */
class Feature{
public:
	Feature(const std::string &featureString);
	Feature(const std::string &name, const std::string &date);
	Feature(
		const std::string &name,
		const std::string &date,
		const std::string &doDescription,
		const std::string &verifyDescription
	);

	void setDate(const std::string &date);

	const std::string& getName() const;
	const std::string getDate() const;
	const std::string& getDoDescription() const;
	const std::string& getVerifyDescription() const;
private:
	std::string name;
	unsigned int date[3];
	std::string doDescription;
	std::string verifyDescription;
};

inline const std::string& Feature::getName() const{
	return name;
}

inline const std::string Feature::getDate() const{
	return std::to_string(date[0])
		+ "-" + std::to_string(date[1])
		+ "-" + std::to_string(date[2]);
}

inline const std::string& Feature::getDoDescription() const{
	return doDescription;
}

inline const std::string& Feature::getVerifyDescription() const{
	return verifyDescription;
}

};	//End of namespace FeatureChecker
};	//End of namespace TBTK

#endif
