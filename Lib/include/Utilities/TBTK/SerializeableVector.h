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
 *  @file SerializeableVector.h
 *  @brief Serializeable wrapper of std::vector.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SERIALIZEABLE_VECTOR
#define COM_DAFER45_TBTK_SERIALIZEABLE_VECTOR

#include "TBTK/Serializeable.h"

#include <string>
#include <vector>

#include "TBTK/json.hpp"

namespace TBTK{

template<typename DataType, bool = std::is_base_of<Serializeable, DataType>::value>
class SerializeableVector : public std::vector<DataType>, Serializeable{
public:
	/** Constructor. */
	SerializeableVector() : std::vector<DataType>(){};

	/** Constructor. */
	SerializeableVector(
		const std::vector<DataType> &v
	) : std::vector<DataType>(v){};

	/** Constructor. Constructs the SerializeableVector from a
	 *  serialization string.*/
	SerializeableVector(const std::string &serialization, Mode mode);

	/** Implements Serializeable::serialize(). */
	std::string serialize(Mode mode) const;
private:
};

template<typename DataType>
class SerializeableVector<DataType, true> :
	public std::vector<DataType>, Serializeable
{
public:
	/** Constructor. */
	SerializeableVector() : std::vector<DataType>(){};

	/** Constructor. */
	SerializeableVector(
		const std::vector<DataType> &v
	) : std::vector<DataType>(v){};

	/** Constructor. Constructs the SerializeableVector from a
	 *  serialization string.*/
	SerializeableVector(const std::string &serialization, Mode mode);

	/** Implements Serializeable::serialize(). */
	std::string serialize(Mode mode) const;
private:
};

template<typename DataType>
class SerializeableVector<DataType, false> :
	public std::vector<DataType>, Serializeable
{
public:
	/** Constructor. */
	SerializeableVector() : std::vector<DataType>(){};

	/** Constructor. */
	SerializeableVector(
		const std::vector<DataType> &v
	) : std::vector<DataType>(v){};

	/** Constructor. Constructs the SerializeableVector from a
	 *  serialization string.*/
	SerializeableVector(const std::string &serialization, Mode mode);

	/** Implements Serializeable::serialize(). */
	std::string serialize(Mode mode) const;
private:
};

template<typename DataType>
SerializeableVector<DataType, false>::SerializeableVector(
	const std::string &serialization,
	Mode mode
){
	TBTKAssert(
		validate(serialization, "SerializeableVector", mode),
		"SerializeableVector::SerializeableVector()",
		"Unable to parse string as SerializeableVector '" << serialization << "'.",
		""
	);

	switch(mode){
	case Mode::JSON:
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			nlohmann::json elements = j.at("elements");
			for(
				nlohmann::json::iterator it = elements.begin();
				it < elements.end();
				++it
			){
				DataType value;
				Serializeable::deserialize(
					*it,
					&value,
					mode
				);
				std::vector<DataType>::push_back(value);
			}
		}
		catch(nlohmann::json::exception e){
			TBTKExit(
				"SerializeableVector::SerializeableVector()",
				"Unable to parse string as SerializeableVector"
				<< " '" << serialization << "'.",
				""
			);
		}

		break;
	default:
		TBTKExit(
			"SerializeableVector::SerializeableVector()",
			"Only SerializeableVector::Mode::JSON is supported yet.",
			""
		);
	}
}

template<typename DataType>
SerializeableVector<DataType, true>::SerializeableVector(
	const std::string &serialization,
	Mode mode
){
	TBTKAssert(
		validate(serialization, "SerializeableVector", mode),
		"SerializeableVector::SerializeableVector()",
		"Unable to parse string as SerializeableVector '" << serialization << "'.",
		""
	);

	switch(mode){
	case Mode::JSON:
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			nlohmann::json elements = j.at("elements");
			for(
				nlohmann::json::iterator it = elements.begin();
				it < elements.end();
				++it
			){
				std::vector<DataType>::push_back(
					DataType(*it, mode)
				);
			}
		}
		catch(nlohmann::json::exception e){
			TBTKExit(
				"SerializeableVector::SerializeableVector()",
				"Unable to parse string as SerializeableVector"
				<< " '" << serialization << "'.",
				""
			);
		}

		break;
	default:
		TBTKExit(
			"SerializeableVector::SerializeableVector()",
			"Only SerializeableVector::Mode::JSON is supported yet.",
			""
		);
	}
}

template<typename DataType>
std::string SerializeableVector<DataType, false>::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "SerializeableVector";
		j["elements"] = nlohmann::json::array();
		for(unsigned int n = 0; n < std::vector<DataType>::size(); n++){
			j["elements"].push_back(
				Serializeable::serialize(
					std::vector<DataType>::at(n),
					mode
				)
			);
		}

		return j.dump();
	}
	default:
		TBTKExit(
			"SerializeableVector::serialize()",
			"Only Serializeable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<typename DataType>
std::string SerializeableVector<DataType, true>::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "SerializeableVector";
		j["elements"] = nlohmann::json::array();
		for(unsigned int n = 0; n < std::vector<DataType>::size(); n++){
			j["elements"].push_back(
				std::vector<DataType>::at(n).serialize(mode)
			);
		}

		return j.dump();
	}
	default:
		TBTKExit(
			"SerializeableVector::serialize()",
			"Only Serializeable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End of namespace TBTK

#endif
