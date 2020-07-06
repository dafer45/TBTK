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
 *  @file Invalidateable.h
 *  @brief Container for objects that can be invalidated.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_INVALIDATABLE
#define COM_DAFER45_TBTK_INVALIDATABLE

#include "TBTK/Serializable.h"
#include "TBTK/TBTKMacros.h"

#include "TBTK/json.hpp"

namespace TBTK{

/** @brief Container for objects that can be invalidated. */
template<typename DataType>
class Invalidatable : public DataType, public Serializable{
public:
	/** Default constructor. */
	Invalidatable();

	/** Constructs an Invalidatable from an object.
	 *
	 *  @param data Data that will be copied to the content of the
	 *  Invalidatable. */
	Invalidatable(const DataType &data);

	/** Constructs a Invalidatable from a serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the Invalidatable.
	 *
	 *  @param mode The mode with which the string has been serialized. */
	Invalidatable(const std::string &serialization, Mode mode);

	/** Assignmanet operator.
	 *
	 *  @param data The data to assign.
	 *
	 *  @return The left hand side after assignment. */
	Invalidatable& operator=(const DataType &data);

	/** Set whether the object is valid or not.
	 *
	 *  @param isValid Flag indicating whether the object is valid or not.
	 */
	void setIsValid(bool isValid);

	/** Get whether the object is valid or not.
	 *
	 *  @return True if the object is marked as valid, otherwise false. */
	bool getIsValid() const;

	/** Implements Serializable::serialize(). */
	std::string serialize(Mode mode) const;
private:
	/** Flag indicating whether the object is marked as valid. */
	bool isValid;
};

template<typename DataType>
Invalidatable<DataType>::Invalidatable(){
}

template<typename DataType>
Invalidatable<DataType>::Invalidatable(const DataType &data) : DataType(data){
}

template<typename DataType>
Invalidatable<DataType>::Invalidatable(
	const std::string &serialization,
	Mode mode
) :
	DataType(Serializable::extract(serialization, mode, "dataType"), mode)
{
	TBTKAssert(
		validate(serialization, "Invalidatable", mode),
		"Invalidatable::Invalidatable()",
		"Unable to parse string as Invalidatable '" << serialization
		<< "'.",
		""
	);
	switch(mode){
	case Mode::JSON:
		try{
			nlohmann::json j
				= nlohmann::json::parse(serialization);
			isValid = j.at("isValid").get<bool>();
		}
		catch(nlohmann::json){
			TBTKExit(
				"Invalidatable::Invalidatable()",
				"Unable to parse string as Invalidatable '"
				<< serialization << "'.",
				""
			);
		}
		break;
	default:
		TBTKExit(
			"Invalidatable::Invalidatable()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<typename DataType>
Invalidatable<DataType>& Invalidatable<DataType>::operator=(
	const DataType &rhs
){
	if(this != &rhs)
		DataType::operator=(rhs);

	return *this;
}

template<typename DataType>
void Invalidatable<DataType>::setIsValid(bool isValid){
	this->isValid = isValid;
}

template<typename DataType>
bool Invalidatable<DataType>::getIsValid() const{
	return isValid;
}

template<typename DataType>
std::string Invalidatable<DataType>::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "Invalidatable";
		j["isValid"] = isValid;
		j["dataType"] = nlohmann::json::parse(
			DataType::serialize(mode)
		);

		return j.dump();
	}
	default:
		TBTKExit(
			"Invalidatable::serialize()",
			"Only Serializable::Mode::JSON implemented yet.",
			""
		);
	}
}

}; //End of namesapce TBTK

#endif
