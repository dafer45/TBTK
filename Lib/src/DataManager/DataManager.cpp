/* Copyright 2017 Kristofer Björnson
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

/** @file DataManager.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/DataManager.h"
#include "TBTK/Resource.h"
#include "TBTK/TBTKMacros.h"

#include <algorithm>

#include "TBTK/json.hpp"

using namespace std;
//using namespace nlohmann;

namespace TBTK{

DataManager::DataManager(
	const vector<double> &lowerBounds,
	const vector<double> &upperBounds,
	const vector<unsigned int> &numTicks,
	const vector<string> &parameterNames,
	const string &dataManagerName
){
	TBTKAssert(
		lowerBounds.size() == upperBounds.size(),
		"DataManager::DataManager()",
		"Incompatible array sizes.",
		"lowerBounds and upperBounds has to have the same number of"
		<< " elements."
	);
	TBTKAssert(
		lowerBounds.size() == numTicks.size(),
		"DataManager::DataManager()",
		"Incompatible array sizes.",
		"lowerBounds and numTicks has to have the same number of"
		<< " elements."
	);
	TBTKAssert(
		lowerBounds.size() == parameterNames.size(),
		"DataManager::DataManager()",
		"Incompatible array sizes.",
		"lowerBounds and parameterNames has to have the same number of"
		<< " elements."
	);
	for(unsigned int n = 0; n < lowerBounds.at(n); n++){
		TBTKAssert(
			lowerBounds.at(n) < upperBounds.at(n),
			"DataManager::DataManager()",
			"Invalid bounds. Lower bound '" << n << "' is smaller"
			<< " than the corresponding upper bound.",
			""
		);
		TBTKAssert(
			numTicks.at(n) > 1,
			"DataManager::DataManager()",
			"Invalid numTicks, '0' ticks is given at position '" << n << "'.",
			""
		);
	}

	this->lowerBounds = lowerBounds;
	this->upperBounds = upperBounds;
	this->numTicks = numTicks;
	this->parameterNames = parameterNames;

	numDataPoints = 1;
	for(unsigned int n = 0; n < numTicks.size(); n++)
		numDataPoints *= numTicks.at(n);

	this->dataManagerName = dataManagerName;

	this->path = "";
}

DataManager::DataManager(const string &serialization, Mode mode){
	TBTKAssert(
		validate(serialization, "DataManager", mode),
		"DataManager::DataManager()",
		"Unable to parse string as DataManager '" << serialization
		<< "'.",
		""
	);

	switch(mode){
	case Mode::JSON:
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);

			nlohmann::json lowerBounds = j.at("lowerBounds");
			for(
				nlohmann::json::iterator it = lowerBounds.begin();
				it < lowerBounds.end();
				++it
			){
				this->lowerBounds.push_back(*it);
			}

			nlohmann::json upperBounds = j.at("upperBounds");
			for(
				nlohmann::json::iterator it = upperBounds.begin();
				it < upperBounds.end();
				++it
			){
				this->upperBounds.push_back(*it);
			}

			nlohmann::json numTicks = j.at("numTicks");
			for(
				nlohmann::json::iterator it = numTicks.begin();
				it < numTicks.end();
				++it
			){
				this->numTicks.push_back(*it);
			}

			nlohmann::json parameterNames = j.at("parameterNames");
			for(
				nlohmann::json::iterator it = parameterNames.begin();
				it < parameterNames.end();
				++it
			){
				this->parameterNames.push_back(*it);
			}

			dataManagerName = j.at("dataManagerName");
			path = j.at("path");

			numDataPoints = j.at("numDataPoints");

			nlohmann::json dataTypes = j.at("dataTypes");
			nlohmann::json reservedDataPoints = j.at("reservedDataPoints");
			nlohmann::json completedDataPoints = j.at("completedDataPoints");
			int counter = 0;
			for(
				nlohmann::json::iterator it = dataTypes.begin();
				it < dataTypes.end();
				++it
			){
				this->dataTypes.push_back(*it);

				bool *reserved = new bool[numDataPoints];
				for(unsigned int n = 0; n < numDataPoints; n++)
					reserved[n] = false;
				int counter2 = 0;
				for(
					nlohmann::json::iterator it2 = reservedDataPoints.at(counter).begin();
					it2 < reservedDataPoints.at(counter).end(); ++it2
				){
					reserved[counter2] = *it2;
					counter2++;
				}
				this->reservedDataPoints.push_back(reserved);

				bool *completed = new bool[numDataPoints];
				for(unsigned int n = 0; n < numDataPoints; n++)
					reserved[n] = false;
				counter2 = 0;
				for(
					nlohmann::json::iterator it2 = completedDataPoints.at(counter).begin();
					it2 < completedDataPoints.at(counter).end(); ++it2
				){
					completed[counter2] = *it2;
					counter2++;
				}
				this->completedDataPoints.push_back(completed);

				counter++;
			}

			nlohmann::json fileTypes = j.at("fileTypes");
			for(
				nlohmann::json::iterator it = fileTypes.begin();
				it < fileTypes.end();
				++it
			){
				this->fileTypes.push_back(*it);
			}
		}
		catch(nlohmann::json::exception &e){
			TBTKExit(
				"DataManager::DataManager()",
				"Unable to parse string as DataManager '"
				<< serialization << "'.",
				""
			);
		}
		break;
	default:
		TBTKExit(
			"DataManager::DataManager()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

DataManager::~DataManager(){
	for(unsigned int n = 0; n < reservedDataPoints.size(); n++)
		delete [] reservedDataPoints.at(n);
	for(unsigned int n = 0; n < completedDataPoints.size(); n++)
		delete [] completedDataPoints.at(n);
}

void DataManager::setPath(const string &path){
	this->path = path;
	if(path.size() > 0){
		if(path.back() != '/')
			this->path += '/';
	}
}

void DataManager::addDataType(const std::string &dataType, FileType fileType){
	for(unsigned int n = 0; n < dataTypes.size(); n++){
		TBTKAssert(
			dataTypes.at(n).compare(dataType) != 0,
			"DataManager::addDataType()",
			"Data type '" << dataType << "' has already been"
			<< " added.",
			""
		);
	}

	dataTypes.push_back(dataType);
	fileTypes.push_back(fileType);
	addDataTables();
}

int DataManager::reserveDataPoint(const std::string &dataType){
	for(unsigned int n = 0; n < numDataPoints; n++){
		if(reserveDataPoint(dataType, n))
			return n;
	}

	return -1;
}

vector<double> DataManager::getParameters(int id) const{
	TBTKAssert(
		id >= 0 && (unsigned int)id < numDataPoints,
		"DataManager::getDataPoint()",
		"The ID is out of range.",
		""
	);

	vector<unsigned int> dataPoint = getDataPoint(id);
	vector<double> parameters;
	for(unsigned int n = 0; n < lowerBounds.size(); n++){
		if(numTicks.at(n) == 1){
			parameters.push_back(lowerBounds.at(n));
		}
		else{
			parameters.push_back(
				lowerBounds.at(n)
				+ dataPoint.at(n)*(upperBounds.at(n) - lowerBounds.at(n))/(numTicks.at(n)-1)
			);
		}
	}

	return parameters;
}

string DataManager::getFilename(const string &dataType, int id) const{
	string filename;
	if(dataManagerName.compare("") != 0){
		filename = dataManagerName + "_";
	}
	filename += to_string(id) + "_" + dataType;

	switch(fileTypes.at(getDataTypeIndex(dataType))){
	case FileType::Custom:
		break;
	case FileType::SerializableJSON:
		filename += ".json";
		break;
	case FileType::PNG:
		filename += ".png";
		break;
	default:
		TBTKExit(
			"DataManager::getFilename()",
			"Unknown data type.",
			"This should never happen, contact the developer."
		);
	}

	return filename;
}

void DataManager::markCompleted(
	const string &dataType,
	int id
){
	TBTKAssert(
		id >= 0 && (unsigned int)id < numDataPoints,
		"DataManager::markCompleted()",
		"The ID is out of range.",
		""
	);

	if(dataType.compare("") == 0){
		for(unsigned int n = 0; n < dataTypes.size(); n++)
			completedDataPoints.at(n)[id] = true;
	}
	else{
		int dataTypeIndex = getDataTypeIndex(dataType);

		completedDataPoints.at(dataTypeIndex)[id] = true;
	}
}

void DataManager::complete(
	const Serializable &serializable,
	const string &dataType,
	int id
){
	TBTKAssert(
		id >= 0 && (unsigned int)id < numDataPoints,
		"DataManager::markCompleted()",
		"The ID is out of range.",
		""
	);

	Resource resource;
	string filename = path + getFilename(dataType, id);
	switch(fileTypes.at(getDataTypeIndex(dataType))){
	case FileType::SerializableJSON:
		resource.setData(serializable.serialize(Mode::JSON));
		break;
	default:
		TBTKExit(
			"DataManager::complete()",
			"Automatic completion is not supported for data type '"
			<< dataType << "'.",
			"Instead use DataManager::getFilename() to retreive a"
			<< " unique filename, save the data manually using"
			<< " this filename, and finally mark the entry"
			<< " completed using DataManager::markCompleted()."
		);
	}

	resource.write(filename);

	markCompleted(dataType, id);
}

void DataManager::addDataTables(){
	bool *reserved = new bool[numDataPoints];
	for(unsigned int n = 0; n < numDataPoints; n++)
		reserved[n] = false;
	reservedDataPoints.push_back(reserved);

	bool *completed = new bool[numDataPoints];
	for(unsigned int n = 0; n < numDataPoints; n++)
		completed[n] = false;
	completedDataPoints.push_back(completed);
}

unsigned int DataManager::getID(const vector<unsigned int> &dataPoint) const{
	TBTKAssert(
		dataPoint.size() == numTicks.size(),
		"DataManager::getID()",
		"Invalid data point size.",
		""
	);

	for(unsigned int n = 0; n < dataPoint.size(); n++){
		TBTKAssert(
			dataPoint.at(n) < numTicks.at(n),
			"DataManager::getID()",
			"dataPoint is out of bound.",
			""
		);
	}

	unsigned int id = 0;
	unsigned int multiplier = 1;
	for(int n = dataPoint.size() - 1; n >= 0; n--){
		id += multiplier*dataPoint.at(n);
		multiplier *= numTicks.at(n);
	}

	return id;
}

vector<unsigned int> DataManager::getDataPoint(unsigned int id) const{
	TBTKAssert(
		id < numDataPoints,
		"DataManager::getDataPoint()",
		"The ID is out of range.",
		""
	);

	vector<unsigned int> dataPoint;
	for(int n = numTicks.size()-1; n >= 0; n--){
		dataPoint.push_back(id%numTicks.at(n));
		id /= numTicks.at(n);
	}

	reverse(dataPoint.begin(), dataPoint.end());
	return dataPoint;
}

unsigned int DataManager::getDataTypeIndex(const string &dataType) const{
	int dataTypeIndex = -1;
	for(unsigned int n = 0; n < dataTypes.size(); n++){
		if(dataTypes.at(n).compare(dataType) == 0){
			dataTypeIndex = n;
			break;
		}
	}
	TBTKAssert(
		dataTypeIndex >= 0,
		"DataManager::getDataTypeIndex()",
		"Data type '" << dataType << "' has not been added to the"
		<< " DataManager.",
		"Use DataManager::addDataType() to add data type."
	);

	return (unsigned int)dataTypeIndex;
}

bool DataManager::reserveDataPoint(
	const string &dataType,
	unsigned int id
){
	if(dataType.compare("") == 0){
		for(unsigned int n = 0; n < dataTypes.size(); n++){
			if(
				completedDataPoints.at(n)[id]
				|| reservedDataPoints.at(n)[id]
			){
				return false;
			}
		}

		for(unsigned int n = 0; n < dataTypes.size(); n++)
			reservedDataPoints.at(n)[id] = true;
		return true;
	}
	else{
		int dataTypeIndex = getDataTypeIndex(dataType);

		if(
			!completedDataPoints.at(dataTypeIndex)[id]
			&& !reservedDataPoints.at(dataTypeIndex)[id]
		){
			reservedDataPoints.at(dataTypeIndex)[id] = true;
			return true;
		}

		return false;
	}
}

string DataManager::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "DataManager";
		for(unsigned int n = 0; n < lowerBounds.size(); n++){
			j["lowerBounds"].push_back(lowerBounds.at(n));
			j["upperBounds"].push_back(upperBounds.at(n));
			j["numTicks"].push_back(numTicks.at(n));
			j["parameterNames"].push_back(parameterNames.at(n));
		}
		j["dataManagerName"] = dataManagerName;
		j["path"] = path;
		j["numDataPoints"] = numDataPoints;
		for(unsigned int n = 0; n < dataTypes.size(); n++){
			j["dataTypes"] = dataTypes;
			j["fileTypes"] = fileTypes;
			j["reservedDataPoints"].push_back(nlohmann::json());
			j["completedDataPoints"].push_back(nlohmann::json());
			for(unsigned int c = 0; c < numDataPoints; c++){
				//Do not export information about reserved
				//data. At least for now reservation is not
				//persistent accross sessions.
				j["reservedDataPoints"].at(n).push_back(false);
				//Completion is preserved.
				j["completedDataPoints"].at(n).push_back(
					completedDataPoints.at(n)[c]
				);
			}
		}

		return j.dump();
	}
	default:
		TBTKExit(
			"DataManager::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End of namespace TBTK
