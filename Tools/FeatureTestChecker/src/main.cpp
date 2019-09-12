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

/** @package TBTKtemp
 *  @file main.cpp
 *  @brief New project
 *
 *  Empty template project.
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/TBTKMacros.h"
#include "TBTK/Streams.h"

#include <complex>
#include <iomanip>
#include <tuple>

using namespace std;
using namespace TBTK;

vector<string> splitString(string str, char delimiter){
	vector<string> components;
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

vector<tuple<string, unsigned int>> getFeatureLines(string filename){
	unsigned int state = 0;
	string TBTK_FEATURE_TEST_ID = "TBTKFeature";

	ifstream fin(filename);
	vector<tuple<string, unsigned int>> featureLines;
	char c;
	unsigned int currentLine = 0;
	while(fin >> c){
		if(c == '\n')
			currentLine++;

		if(c == TBTK_FEATURE_TEST_ID[state])
			state++;
		else
			state = 0;

		if(state == TBTK_FEATURE_TEST_ID.size()){
			string line;
			getline(fin, line);
			featureLines.push_back(make_tuple("TBTKFeature " + line, currentLine));
			state = 0;
		}
	}

	return featureLines;
}

vector<string> getTestNames(
	const vector<tuple<string, unsigned int>> &featureLines
){
	vector<string> names;
	for(unsigned int n = 0; n < featureLines.size(); n++){
		vector<string> components
			= splitString(get<0>(featureLines[n]), ' ');
		TBTKAssert(
			components.size() == 3,
			"TBTKFeatureChecker",
			"Encountered error while parsing line '"
			<< get<1>(featureLines[n]) << "': "
			<< get<0>(featureLines[n]) << ".",
			"The format should be TBTKFeature"
			<< " Class.Function.Test YYYY-MM-DD."
		);

		names.push_back(components[1]);
	}

	return names;
}

vector<vector<unsigned int>> getTestDates(
	const vector<tuple<string, unsigned int>> &featureLines
){
	vector<vector<unsigned int>> dates;
	for(unsigned int n = 0; n < featureLines.size(); n++){
		vector<string> components
			= splitString(get<0>(featureLines[n]), ' ');
		TBTKAssert(
			components.size() == 3,
			"TBTKFeatureChecker",
			"Encountered error while parsing line '"
			<< get<1>(featureLines[n]) << "': "
			<< get<0>(featureLines[n]) << ".",
			"The format should be TBTKFeature"
			<< " Class.Function.Test YYYY-MM-DD."
		);

		string date = components[2];

		vector<string> dateComponents = splitString(date, '-');
		TBTKAssert(
			dateComponents.size() == 3,
			"TBTKFeatureChecker",
			"Encountered error while parsing line '"
			<< get<1>(featureLines[n]) << "': "
			<< get<0>(featureLines[n]) << ".",
			"The format should be TBTKFeature"
			<< " Class.Function.Test YYYY-MM-DD."
		);

		dates.push_back(vector<unsigned int>());
		for(unsigned int n = 0; n < dateComponents.size(); n++)
			dates.back().push_back(stoi(dateComponents[n]));
	}

	return dates;
}

vector<unsigned int> getTestIds(const vector<string> &names, const string &name){
	vector<unsigned int> ids;
	for(unsigned n = 0; n < names.size(); n++)
		if(names[n].compare(name) == 0)
			ids.push_back(n);

	return ids;
}

bool getIsUpToDate(
	const vector<unsigned int> &date0,
	const vector<unsigned int> &date1
){
	for(unsigned int n = 0; n < 3; n++){
		if(date0[n] > date1[n])
			return false;
	}

	return true;
}

int main(int argc, char **argv){
	TBTKAssert(
		argc == 3,
		"TBTKFeatureChecker",
		"Please pass the reference as first argument, and the test"
		<< " suite as the second argument.",
		""
	);

	vector<tuple<string, unsigned int>> featureLines[2];
	vector<string> names[2];
	vector<vector<unsigned int>> dates[2];
	for(unsigned int n = 0; n < 2; n++){
		featureLines[n] = getFeatureLines(argv[n+1]);
		names[n] = getTestNames(featureLines[n]);
		dates[n] = getTestDates(featureLines[n]);
	}

	Streams::out << left << setw(40) << "Test name" << setw(10) << "Exists" << setw(15) << "Up to date" << setw(15) << "Multiplicity" << "\n";
	for(unsigned int n = 0; n < names[0].size(); n++){
		const string &name = names[0][n];
		Streams::out << setw(40) << name;

		vector<unsigned int> ids = getTestIds(names[1], name);
		if(ids.size() != 0){
			Streams::out << setw(10) << "X";

			bool isUpToDate = true;
			for(unsigned int c = 0; c < ids.size(); c++){
				if(!getIsUpToDate(dates[0][n], dates[1][ids[c]]))
					isUpToDate = false;
			}
			if(isUpToDate)
				Streams::out << setw(15) << "X";
			else
				Streams::out << setw(15) << "";
			Streams::out << setw(15) << ids.size();
		}

		Streams::out << "\n";
	}

	vector<string> unknownFeatures;
	for(unsigned int n = 0; n < names[1].size(); n++){
		const string &name = names[1][n];

		vector<unsigned int> ids = getTestIds(names[0], name);
		if(ids.size() == 0)
			unknownFeatures.push_back(name);
	}

	if(unknownFeatures.size() != 0){
		Streams::out << "\nUnknown features (possibly removed features):\n";
		for(unsigned int n = 0; n < unknownFeatures.size(); n++)
			Streams::out << unknownFeatures[n] << "\n";
	}

	return 0;
}
