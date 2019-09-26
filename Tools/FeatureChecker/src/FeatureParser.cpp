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
#include "FeatureParser.h"
#include "TBTK/TBTKMacros.h"

#include <sstream>

#include "TBTK/json.hpp"

using namespace std;
using namespace nlohmann;

namespace TBTK{
namespace FeatureChecker{

vector<Feature> FeatureParser::parseJSON(const string &filename){
	json j = openJSON(filename);
	vector<Feature> features;
	json jsonFeatures = j.at("Features");
	for(
		json::iterator iterator = jsonFeatures.begin();
		iterator != jsonFeatures.end();
		++iterator
	){
		string name = iterator->at("Name");
		string date = iterator->at("Date");
		string doDescription = iterator->at("Do");
		string verifyDescription = iterator->at("Verify");
		features.push_back(
			Feature(name, date, doDescription, verifyDescription)
		);
	}

	return features;
}

vector<Feature> FeatureParser::parseSourceFile(const string &filename){
	vector<string> featureLines = extractFeatureLines(filename);

	vector<Feature> features;
	for(unsigned int n = 0; n < featureLines.size(); n++)
		features.push_back(Feature(featureLines[n]));

	return features;
}

json FeatureParser::openJSON(const string &filename){
	ifstream fin(filename);
	TBTKAssert(
		fin.is_open(),
		"FeatureParser::parseJSON()",
		"Unable to open file '" << filename << "'.",
		""
	);
	stringstream ss;
	ss << fin.rdbuf();
	json j = json::parse(ss.str());

	return j;
}

vector<string> FeatureParser::extractFeatureLines(const string &filename){
	unsigned int state = 0;
	string TBTK_FEATURE_ID = "TBTKFeature";

	ifstream fin(filename);
	TBTKAssert(
		fin.is_open(),
		"FeatureParser::extractFeatureLines()",
		"Unable to open file '" << filename << "'.",
		""
	);

	vector<string> featureLines;
	char c;
	while(fin >> c){
		if(c == TBTK_FEATURE_ID[state])
			state++;
		else
			state = 0;

		if(state == TBTK_FEATURE_ID.size()){
			string line;
			getline(fin, line);
			featureLines.push_back("TBTKFeature " + line);
			state = 0;
		}
	}
	fin.close();

	return featureLines;
}

};
};
