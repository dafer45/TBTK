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

/** @file Feature.cpp
 *  @author Kristofer Björnson
 */

#include "Feature.h"
#include "TBTK/TBTKMacros.h"
#include "Utilities.h"

using namespace std;

namespace TBTK{
namespace FeatureChecker{

Feature::Feature(const string &featureString){
	vector<string> components = splitString(featureString, ' ');
	TBTKAssert(
		components.size() == 3,
		"Feature::Feature()",
		"Unable to parse '" << featureString << "' as a Feature"
		<< " string.",
		""
	);
	name = components[1];
	setDate(components[2]);
}

Feature::Feature(const string &name, const string &date){
	this->name = name;
	setDate(date);
}

Feature::Feature(
	const string &name,
	const string &date,
	const string &doDescription,
	const string &verifyDescription
){
	this->name = name;
	setDate(date);
	this->doDescription = doDescription;
	this->verifyDescription = verifyDescription;
}

void Feature::setDate(const string &date){
	vector<string> components = splitString(date, '-');
	TBTKAssert(
		components.size() == 3,
		"Feature::setDate()",
		"Invalid date string '" << date << "'.",
		""
	);
	for(unsigned int n = 0; n < 3; n++)
		this->date[n] = atoi(components[n].c_str());
}

};
};
