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

/** @package TBTKcalc
 *  @file DataManager.h
 *  @brief Manages data.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_DATA_MANAGER
#define COM_DAFER45_TBTK_DATA_MANAGER

#include "TBTK/Serializeable.h"

#include <string>
#include <vector>

namespace TBTK{

class DataManager : public Serializeable{
public:
	/** Constructor. */
	DataManager(
		const std::vector<double> &lowerBounds,
		const std::vector<double> &upperBounds,
		const std::vector<unsigned int> &numTicks,
		const std::vector<std::string> &parameterNames,
		const std::string &dataManagerName = ""
	);

	/** Copy constructor. */
	DataManager(const DataManager &dataManager) = delete;

	/** Constructor. Constructs the DataManager from a serialization
	 *  string. */
	DataManager(const std::string &serialization, Mode mode);

	/** Destructor. */
	virtual ~DataManager();

	/** Assignment operator. */
	DataManager& operator=(const DataManager &dataManager) = delete;

	/** Get lower bound. */
	double getLowerBound(unsigned int parameterIndex) const;

	/** Get upper bound. */
	double getUpperBound(unsigned int parameterIndex) const;

	/** Get number of ticks. */
	unsigned int getNumTicks(unsigned int parameterIndex) const;

	/** Get number of parameters. */
	unsigned int getNumParameters() const;

	/** Get parameter name. */
	const std::string& getParameterName(unsigned int parameterIndex) const;

	/** Set output path. */
	void setPath(const std::string &path = "");

	/** Get output path. */
	const std::string& getPath() const;

	enum class FileType {
		Custom,
		SerializeableJSON,
		PNG
	};

	/** Add a data type to manage. */
	void addDataType(
		const std::string &dataType,
		FileType fileType
	);

	/** Get number of data types. */
	unsigned int getNumDataTypes() const;

	/** Get data type. */
	const std::string& getDataType(unsigned int dataTypeIndex) const;

	/** Get file type. */
	FileType getFileType(const std::string &dataType) const;

	/** Reserve a data point. */
	int reserveDataPoint(const std::string &dataTypes = "");

	/** Get parameter for a given ID. */
	std::vector<double> getParameters(int id) const;

	/** Convert a data point to an ID. */
	unsigned int getID(const std::vector<unsigned int> &dataPoint) const;

	/** Convert ID to data point. */
	std::vector<unsigned int> getDataPoint(unsigned int id) const;

	/** Get a (for this DataManager) unique filename that can be used to
	 *  store and retreive a result. */
	std::string getFilename(const std::string &dataType, int id) const;

	/** Mark data point completed. */
	void markCompleted(
		const std::string &dataType,
		int id
	);

	/** Complete save a Serializeable result and mark it as completed. */
	void complete(
		const Serializeable &serializeable,
		const std::string &dataType,
		int id
	);

	/** Implements Serializeable::serialize(). */
	virtual std::string serialize(Mode mode) const;
private:
	/** Lower bounds. */
	std::vector<double> lowerBounds;

	/** Upper bounds. */
	std::vector<double> upperBounds;

	/** Number of ticks. */
	std::vector<unsigned int> numTicks;

	/** Parameter names. */
	std::vector<std::string> parameterNames;

	/** Data manager name. */
	std::string dataManagerName;

	/** Output path. */
	std::string path;

	/** Number of data points. */
	unsigned int numDataPoints;

	/** Data names. */
	std::vector<std::string> dataTypes;

	/** File tpyes. */
	std::vector<FileType> fileTypes;

	/** Table of data points that have been reserved. */
	std::vector<bool*> reservedDataPoints;

	/** Table of data points that have been completed. */
	std::vector<bool*> completedDataPoints;

	/** Add data tables. */
	void addDataTables();

	/** Get data type index. */
	unsigned int getDataTypeIndex(const std::string &dataType) const;

	/** Reserve data point. */
	bool reserveDataPoint(
		const std::string &dataType,
		unsigned int id
	);
};

inline double DataManager::getLowerBound(unsigned int parameterIndex) const{
	return lowerBounds.at(parameterIndex);
}

inline double DataManager::getUpperBound(unsigned int parameterIndex) const{
	return upperBounds.at(parameterIndex);
}

inline unsigned int DataManager::getNumTicks(unsigned int parameterIndex) const{
	return numTicks.at(parameterIndex);
}

inline unsigned int DataManager::getNumParameters() const{
	return parameterNames.size();
}

inline const std::string& DataManager::getParameterName(
	unsigned int parameterIndex
) const{
	return parameterNames.at(parameterIndex);
}

inline unsigned int DataManager::getNumDataTypes() const{
	return dataTypes.size();
}

inline const std::string& DataManager::getDataType(
	unsigned int dataTypeIndex
) const{
	return dataTypes.at(dataTypeIndex);
}

inline DataManager::FileType DataManager::getFileType(
	const std::string &dataType
) const{
	return fileTypes.at(getDataTypeIndex(dataType));
}

inline const std::string& DataManager::getPath() const{
	return path;
}

};	//End namespace TBTK

#endif
