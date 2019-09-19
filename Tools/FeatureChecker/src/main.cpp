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

#include <algorithm>
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
	TBTKAssert(
		fin.is_open(),
		"TBTKFeatureChecker::getFeatureLines()",
		"Unable to open file '" << filename << "'. No such file exists.",
		""
	);
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

vector<tuple<string, bool, bool, unsigned int>> executeFeatureCheck(const vector<string> filenames){
	TBTKAssert(
		filenames.size() == 2,
		"TBTKFeatureChecker::excuteFeatureCheck()",
		"Expected 2 arguments, but received " << filenames.size() << ".",
		""
	);

	vector<tuple<string, unsigned int>> featureLines[2];
	vector<string> names[2];
	vector<vector<unsigned int>> dates[2];
	for(unsigned int n = 0; n < 2; n++){
		featureLines[n] = getFeatureLines(filenames[n]);
		names[n] = getTestNames(featureLines[n]);
		dates[n] = getTestDates(featureLines[n]);
	}

	vector<tuple<string, bool, bool, unsigned int>> results;

	for(unsigned int n = 0; n < names[0].size(); n++){
		const string &name = names[0][n];
		bool exists = false;
		bool isUpToDate = false;
		unsigned int multiplicity;

		vector<unsigned int> ids = getTestIds(names[1], name);
		if(ids.size() != 0){
			exists = true;

			isUpToDate = true;
			for(unsigned int c = 0; c < ids.size(); c++){
				if(!getIsUpToDate(dates[0][n], dates[1][ids[c]]))
					isUpToDate = false;
			}
		}
		multiplicity = ids.size();


		results.push_back(make_tuple(name, exists, isUpToDate, multiplicity));
	}

	return results;
}

void printFeatureCheckResults(
	const vector<tuple<string, bool, bool, unsigned int>> &results
){
	Streams::out << left
		<< setw(40) << "Feature name"
		<< setw(10) << "Exists"
		<< setw(15) << "Up to date"
		<< setw(15) << "Multiplicity" << "\n";
	for(unsigned int n = 0; n < results.size(); n++){
		Streams::out << setw(40) << get<0>(results[n]);
		if(get<1>(results[n])){
			Streams::out << setw(10) << "X";
			if(get<2>(results[n]))
				Streams::out << setw(15) << "X";
			else
				Streams::out << setw(15) << "";
			Streams::out << setw(15) << get<3>(results[n]);
		}
		Streams::out << "\n";
	}
}

unsigned int getNumImplementedFeatures(
	const vector<tuple<string, bool, bool, unsigned int>> &results
){
	unsigned int numImplementedFeatures = 0;
	for(unsigned int n = 0; n < results.size(); n++)
		if(get<1>(results[n]))
			numImplementedFeatures++;

	return numImplementedFeatures;
}

unsigned int getNumUpToDateFeatures(
	const vector<tuple<string, bool, bool, unsigned int>> &results
){
	unsigned int numUpToDateFeatures = 0;
	for(unsigned int n = 0; n < results.size(); n++)
		if(get<2>(results[n]))
			numUpToDateFeatures++;

	return numUpToDateFeatures;
}

void executeReverseFeatureCheck(const vector<string> &filenames){
	TBTKAssert(
		filenames.size() == 2,
		"TBTKFeatureChecker::excuteFeatureCheck()",
		"Expected 2 arguments, but received " << filenames.size() << ".",
		""
	);

	vector<tuple<string, unsigned int>> featureLines[2];
	vector<string> names[2];
	vector<vector<unsigned int>> dates[2];
	for(unsigned int n = 0; n < 2; n++){
		featureLines[n] = getFeatureLines(filenames[n]);
		names[n] = getTestNames(featureLines[n]);
		dates[n] = getTestDates(featureLines[n]);
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
}

string getFeaturesFilename(const string &component){
	vector<string> components = splitString(component, '.');
	string filename = TBTK_RESOURCE_PATH;
	for(unsigned int n = 0; n < components.size(); n++)
		filename += "/" + components[n];
	filename += ".features";

	return filename;
}

vector<tuple<string, string>> getComponentList(const string &filename){
	ifstream fin(filename);
	TBTKAssert(
		fin.is_open(),
		"TBTKFeatureChecker::getComponentsList()",
		"Unable to open '" << filename << "'",
		""
	);

	string path = filename.substr(0, filename.find_last_of("/") + 1);

	vector<tuple<string, string>> components;
	string line;
	while(getline(fin, line)){
		line.erase(
			remove_if(line.begin(), line.end(), ::isspace),
			line.end()
		);
		vector<string> words = splitString(line, '=');
		TBTKAssert(
			words.size() == 2,
			"TBTKFeatureChecker::getComponentsList()",
			"Unable to parse '" << line << "' in '" << filename << "'.",
			""
		);

		components.push_back(make_tuple(words[0], path + words[1]));
	}

	return components;
}

vector<unsigned int> getComponentIds(
	const vector<tuple<string, string>> &components,
	const string &name
){
	vector<unsigned int> ids;
	for(unsigned int n = 0; n < components.size(); n++){
		if(get<0>(components[n]).compare(name) == 0)
			ids.push_back(n);
	}

	return ids;
}

void checkAllComponents(const string &filename){
	vector<tuple<string, string>> referenceComponents = getComponentList(
		string(TBTK_RESOURCE_PATH) + "/all.components"
	);
	vector<tuple<string, string>> components = getComponentList(filename);

	Streams::out << left
		<< setw(29) << "Component" << "|"
		<< setw(8) << "Exists" << "|"
		<< setw(12) << "Implemented" << setw(8) << right << "(%) "
			<< left << "|"
		<< setw(12) << "Up to date" << setw(8) << right << "(%)"
			<< left
		<< "\n";
	for(unsigned int n = 0; n < referenceComponents.size(); n++){
		string componentName = get<0>(referenceComponents[n]);
		vector<unsigned int> ids = getComponentIds(components, componentName);
		Streams::out << setw(29) << componentName << "|";
		switch(ids.size()){
		case 0:
			Streams::out << setw(8) << "" << "|"
				<< setw(20) << "" << "|"
				<< setw(20) << ""
				<< "\n";
			break;
		case 1:
		{
			Streams::out << setw(8) << "X" << "|";

			vector<tuple<string, bool, bool, unsigned int>> results
				= executeFeatureCheck({
					get<1>(referenceComponents[n]),
					get<1>(components[n])
				});

			unsigned int numImplementedFeatures
				= getNumImplementedFeatures(results);
			Streams::out
				<< right << setw(5) << numImplementedFeatures
				<< setw(3) << " / "
				<< left << setw(5) << results.size();

			if(results.size() == 0){
				Streams::out << setw(2) << " (" << setw(3)
					<< 100 << setw(2) << ")" << "|";
			}
			else{
				Streams::out << setprecision(0) << fixed;
				Streams::out << setw(2) << " (" << setw(3)
					<< right << 100*(
						numImplementedFeatures/(double)results.size()
					) << left << setw(2) << ")" << "|";
			}

			unsigned int numUpToDateFeatures
				= getNumUpToDateFeatures(results);
			Streams::out
				<< right << setw(5) << numUpToDateFeatures
				<< setw(3) << " / "
				<< left << setw(5) << results.size();

			if(results.size() == 0){
				Streams::out << setw(3) << "  (" << setw(3)
					<< 100 << setw(2) << ")";
			}
			else{
				Streams::out << setprecision(0) << fixed;
				Streams::out << setw(3) << "  (" << setw(3)
					<< right << 100*(
						numUpToDateFeatures/(double)results.size()
					) << left << setw(1) << ")";
			}

			break;
		}
		default:
			TBTKExit(
				"TBTKFeatureChecker::checkAllComponents()",
				"Multiple occurances of the component '"
				<< componentName << "' in '" << filename
				<< "'.",
				""
			);
		}

		Streams::out << "\n";
	}
}

int main(int argc, char **argv){
	switch(argc){
	case 2:
	{
		string filename = argv[1];
		vector<string> components = splitString(filename, '.');
		TBTKAssert(
			components.size() > 1,
			"TBTKFeatureChecker",
			"Unsupported argument.",
			"Examples of valid uses are 'TBTKFeatureChecker"
			<< " componentListFile.components' and"
			<< " 'TBTKFeatureChecker Core.Index Index.h'."
		);
		TBTKAssert(
			components.back().compare("components") == 0,
			"TBTKFeatureChecker",
			"Unsupported argument.",
			"Examples of valid uses are 'TBTKFeatureChecker"
			<< " componentListFile.components' and"
			<< " 'TBTKFeatureChecker Core.Index Index.h'."
		);

		checkAllComponents(argv[1]);

		break;
	}
	case 3:
	{
		string referenceFile = getFeaturesFilename(argv[1]);
		vector<tuple<string, bool, bool, unsigned int>> results
			= executeFeatureCheck({referenceFile, argv[2]});
		printFeatureCheckResults(results);
		executeReverseFeatureCheck({referenceFile, argv[2]});

		break;
	}
	default:
		TBTKExit(
			"TBTKFeatureChecker",
			"Unsupported number of arguments.",
			"Examples of valid uses are 'TBTKFeatureChecker"
			<< " componentListFile.components' and"
			<< " 'TBTKFeatureChecker Core.Index Index.h'."
		);
	}

	return 0;
}
